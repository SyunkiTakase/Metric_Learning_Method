import os
import csv
import shutil
import argparse
import pandas as pd
import importlib.util
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from method_config import build_metric_method

def load_config(config_path):

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.config

def save_to_csv(csv_file, epoch, train_loss=None, ce_loss=None, metric_loss=None, train_acc=None, val_loss=None, val_acc=None):

    new_row = [
        epoch,
        train_loss if train_loss is not None else '',
        ce_loss if ce_loss is not None else '',
        metric_loss if metric_loss is not None else '',
        train_acc if train_acc is not None else '',
        val_loss if val_loss is not None else '',
        val_acc if val_acc is not None else ''
    ]
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)

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

    # 出力を保存するディレクトリ作成
    base_path = './output/' + method +'/' + str(timestamp) + '/'
    sub_dirs = ['model/', 'log/', 'map/']
    for sub in sub_dirs:
        path = base_path + sub
        os.makedirs(path, exist_ok=True)

    csv_file = base_path + 'log/log.csv'
    if not os.path.exists(csv_file):
        # ヘッダーを作成
        with open(csv_file, mode='w', newline='') as f:
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
        val_loss, val_count = validation_fn(
            device, val_loader, model, ce_loss_func, metric_loss_func
        )

        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Loss: {train_loss/len(train_loader):.4f}, CE Loss: {ce_loss/len(train_loader):.4f}, Metric Loss: {metric_loss/len(train_loader):.4f} ')
        print(f'Epoch [{epoch+1}/{num_epoch}], Validation Loss: {val_loss/len(val_loader):.4f} ')
        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Acc: {train_count/len(train_loader.dataset):.4f}, Validation Acc: {val_count/len(val_loader.dataset):.4f} ')

        # ログとモデルを保存
        save_to_csv(
            csv_file, epoch, train_loss/len(train_loader), ce_loss/len(train_loader), metric_loss/len(train_loader),
                        train_count/len(train_loader.dataset), val_loss/len(val_loader), val_count/len(val_loader.dataset)
        )
        if (epoch+1) % 1 == 0:
            print('saved!!')
            save_model_path = base_path + 'model/' + str(epoch + 1) + '.tar'
            torch.save({
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch
            }, save_model_path)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Path to config Python file")

    args = parser.parse_args()
    config = load_config(args.config_path)

    main(args.config_path, config)
