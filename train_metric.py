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
from utils import load_config, save_model, save_log, save_map

def main(config_path, config):

    timestamp = datetime.now().strftime('%Y%m%d%H')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ハイパーパラメータ
    num_epoch = config['epoch']
    batch_size = config['batch_size']
    lr = config['lr']
    weight_decay = config['weight_decay']
    optim_momentum = config['momentum']
    img_size = config['img_size']
    dataset_name = config['dataset']
    margin = config['margin']
    scale = config['scale']
    method = config['method']
    arch = config['arch']
    _lambda = config['_lambda']
    num_dim = config['num_dim']
    use_hard_triplets = config['hard_triplets']
    easy_margin = config['easy_margin']
    plot_map = config["plot_map"]

    # 出力を保存するディレクトリ作成
    base_path = './output/' + method +'/' + str(timestamp) + '/'
    sub_dirs = ['model/', 'log/', 'map/']
    for sub in sub_dirs:
        path = base_path + sub
        os.makedirs(path, exist_ok=True)

    model_path = base_path + 'model/' 
    log_path = base_path + 'log/log.csv'
    map_path = base_path + 'map/'

    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'CE Loss', 'Metric Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

    cfg_name = config_path.split('/')[-1]
    cfg_dest = f"{base_path}/{cfg_name}"
    shutil.copy(config_path, cfg_dest)

    mean = [0.4915, 0.4823, 0.4468]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])
    
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=test_transform, download=False)

    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100("./data", train=False, transform=test_transform, download=False)

    class_names = train_dataset.classes        
    print('Class Names:', class_names)
    print('Number Of Class:', len(class_names))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # モデルの用意
    model, metric_loss_func, optimizer, train_fn, validation_fn = build_metric_method(
        method, arch, lr, weight_decay, optim_momentum, len(class_names), num_dim, margin, scale, use_hard_triplets, easy_margin, device
    )
    ce_loss_func = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print(model)
    print(metric_loss_func)

    # 学習対象のパラメータを可視化
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {param_count}')

    for epoch in range(num_epoch):
        ce_loss, metric_loss, train_loss, train_count = train_fn(
            device, train_loader, model, optimizer, scaler, epoch, ce_loss_func, metric_loss_func, _lambda
        )
        val_loss, val_count, features_np, labels_np = validation_fn(
            device, val_loader, model, ce_loss_func, metric_loss_func
        )

        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Loss: {train_loss/len(train_loader):.4f}, CE Loss: {ce_loss/len(train_loader):.4f}, Metric Loss: {metric_loss/len(train_loader):.4f} ')
        print(f'Epoch [{epoch+1}/{num_epoch}], Validation Loss: {val_loss/len(val_loader):.4f} ')
        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Acc: {train_count/len(train_loader.dataset):.4f}, Validation Acc: {val_count/len(val_loader.dataset):.4f} ')

        # ログ，モデル，t-SNEを保存
        if (epoch+1) % 1 == 0:
            save_model_path = model_path + str(epoch + 1) + '.tar'
            save_model(model, optimizer, epoch, save_model_path)

        save_log(
            log_path, epoch, train_loss/len(train_loader), ce_loss/len(train_loader), metric_loss/len(train_loader),
                        train_count/len(train_loader.dataset), val_loss/len(val_loader), val_count/len(val_loader.dataset)
        )

        if plot_map:
            save_map_path = map_path + str(epoch + 1) + '.png'
            save_map(features_np, labels_np, class_names, save_map_path)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Path to config Python file")

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args.config_path, config)
