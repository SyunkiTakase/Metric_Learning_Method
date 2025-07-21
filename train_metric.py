import os
import csv
import shutil
import argparse
import pandas as pd
import importlib.util
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from method_config import build_metric_method
from utils import load_config, save_model, save_log, save_featspace

def main(config_path, config):
    """
    メイン関数

    Parameters
    ----------
    config_path : str
        設定ファイルのパス
    config : dict
        設定内容を含む辞書
    Returns
    -------
    None
    """
    timestamp = datetime.now().strftime('%Y%m%d%H') # タイムスタンプの生成
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスの設定
    
    # ハイパーパラメータ
    num_epoch = config['epoch'] # エポック数
    batch_size = config['batch_size'] # バッチサイズ
    lr = config['lr'] # 学習率
    weight_decay = config['weight_decay'] # 重み減衰
    optim_momentum = config['momentum'] # オプティマイザのモメンタム
    img_size = config['img_size'] # 入力画像のサイズ
    dataset_name = config['dataset'] # データセット名
    margin = config['margin'] # マージンの値
    scale = config['scale'] # スケーリングの値
    method = config['method'] # 使用するメトリック学習手法
    arch = config['arch'] # 使用するアーキテクチャ
    _lambda = config['_lambda'] # メトリック学習の重み
    num_dim = config['num_dim'] # 特徴量の次元数
    use_hard_triplets = config['hard_triplets'] # ハードトリプレットを使用するかどうか
    easy_margin = config['easy_margin'] # イージーマージンを使用するかどうか
    vis_featspace = config["vis_featspace"] # 特徴空間の可視化を行うかどうか

    # 出力を保存するディレクトリ作成
    base_path = './output/' + method +'/' + str(timestamp) + '/' # 出力ディレクトリのパス
    sub_dirs = ['model/', 'log/', 'map/'] # サブディレクトリのリスト
    for sub in sub_dirs:
        path = base_path + sub
        os.makedirs(path, exist_ok=True)

    model_path = base_path + 'model/' 
    log_path = base_path + 'log/log.csv'
    featspace_path = base_path + 'featspace/'

    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'CE Loss', 'Metric Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

    cfg_name = config_path.split('/')[-1]
    cfg_dest = f"{base_path}/{cfg_name}"
    shutil.copy(config_path, cfg_dest)

    # データ拡張とデータローダーの設定
    mean = [0.4915, 0.4823, 0.4468]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([ # データ拡張の定義
        transforms.ToTensor(),
        # transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([ # テストデータの前処理
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])
    
    if dataset_name == 'cifar10': # CIFAR-10データセットを使用する場合
        train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=test_transform, download=False)

    elif dataset_name == 'cifar100': # CIFAR-100データセットを使用する場合
        train_dataset = torchvision.datasets.CIFAR100("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100("./data", train=False, transform=test_transform, download=False)

    class_names = train_dataset.classes # クラス名の取得
    print('Class Names:', class_names) 
    print('Number Of Class:', len(class_names))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, # 訓練データローダーの設定
                                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, # 検証データローダーの設定
                                            shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # モデルの用意
    model, metric_loss_func, optimizer, train_fn, validation_fn = build_metric_method( # メトリック学習手法の構築
        method, arch, lr, weight_decay, optim_momentum, len(class_names), num_dim, margin, scale, use_hard_triplets, easy_margin, device
    )
    ce_loss_func = torch.nn.CrossEntropyLoss() # クロスエントロピー損失関数の定義
    scaler = torch.cuda.amp.GradScaler(enabled=True) # 自動混合精度のスケーラー

    print(model)
    print(metric_loss_func)

    # 学習対象のパラメータを可視化
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {param_count}')

    # 学習 & 検証ループ
    for epoch in range(num_epoch):
        ce_loss, metric_loss, train_loss, train_count = train_fn( # 学習関数の呼び出し
            device, train_loader, model, optimizer, scaler, epoch, ce_loss_func, metric_loss_func, _lambda
        )
        val_loss, val_count, features_np, labels_np = validation_fn( # 検証関数の呼び出し
            device, val_loader, model, ce_loss_func, metric_loss_func
        )

        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Loss: {train_loss/len(train_loader):.4f}, CE Loss: {ce_loss/len(train_loader):.4f}, Metric Loss: {metric_loss/len(train_loader):.4f} ')
        print(f'Epoch [{epoch+1}/{num_epoch}], Validation Loss: {val_loss/len(val_loader):.4f} ')
        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Acc: {train_count/len(train_loader.dataset):.4f}, Validation Acc: {val_count/len(val_loader.dataset):.4f} ')

        # ログ，モデル，t-SNEを保存
        if (epoch+1) % 10 == 0:
            save_model_path = model_path + str(epoch + 1) + '.tar' # モデルの保存パス
            save_model(model, optimizer, epoch, save_model_path) # モデルの保存

        save_log( # ログの保存
            log_path, epoch, 
            train_loss/len(train_loader), ce_loss/len(train_loader), metric_loss/len(train_loader),train_count/len(train_loader.dataset), 
            val_loss/len(val_loader), val_count/len(val_loader.dataset)
        )

        if vis_featspace: # 特徴空間の可視化を行う場合
            save_featspace_path = featspace_path + str(epoch + 1) + '.png' # 特徴空間の保存パス
            save_featspace(features_np, labels_np, class_names, save_featspace_path) # 特徴空間の保存

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Path to config Python file")

    args = parser.parse_args()
    config = load_config(args.config_path) # 設定ファイルの読み込み

    main(args.config_path, config)
