import os
import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

def train(device, train_loader, model, criterion, optimizer, scaler, epoch):
    """
    学習用の関数

    Parameters
    ----------
    device : torch.device
        使用するデバイス（CPUまたはGPU）
    train_loader : torch.utils.data.DataLoader
        訓練データローダー
    model : nn.Module
        学習するモデル
    criterion : nn.Module
        損失関数
    optimizer : torch.optim.Optimizer
        モデルのオプティマイザ
    scaler : torch.cuda.amp.GradScaler
        自動混合精度のスケーラー
    epoch : int
        現在のエポック数
    Returns
    -------
    sum_loss : float
        訓練データに対する損失の合計
    count : int
        正しく分類されたサンプルの数
    """
    model.train()
    
    sum_loss = 0.0
    count = 0

    for idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device, non_blocking=True).float() # 画像をデバイスに転送
        labels = labels.to(device, non_blocking=True).long() # ラベルをデバイスに転送
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logits = model(imgs) # モデルの順伝播
                loss = criterion(logits, labels) # 損失の計算
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logits.argmax(dim=1) == labels).item()
        
    return sum_loss, count

def train_metric(device, train_loader, model, optimizer, scaler, epoch, ce_loss_func, metric_loss_func, _lambda):
    """
    学習用の関数（メトリック学習用）

    Parameters
    ----------
    device : torch.device
        使用するデバイス（CPUまたはGPU）
    train_loader : torch.utils.data.DataLoader
        訓練データローダー
    model : nn.Module
        学習するモデル
    optimizer : torch.optim.Optimizer
        モデルのオプティマイザ
    scaler : torch.cuda.amp.GradScaler
        自動混合精度のスケーラー
    epoch : int
        現在のエポック数
    ce_loss_func : nn.Module
        クロスエントロピー損失関数
    metric_loss_func : nn.Module
        メトリック学習の損失関数
    _lambda : float
        メトリック学習の損失関数に対する重み
    Returns
    -------
    sum_ce_loss : float
        クロスエントロピー損失の合計
    sum_metric_loss : float
        メトリック学習の損失の合計
    sum_loss : float
        総損失の合計
    count : int
        正しく分類されたサンプルの数
    """
    model.train()
    
    sum_ce_loss = 0.0
    sum_metric_loss = 0.0
    sum_loss = 0.0
    count = 0

    for idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device, non_blocking=True).float() # 画像をデバイスに転送
        labels = labels.to(device, non_blocking=True).long() # ラベルをデバイスに転送

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features, logits = model(imgs) # モデルの順伝播
            ce_loss = ce_loss_func(logits, labels) # クロスエントロピー損失の計算
            metric_loss = _lambda * metric_loss_func(features, labels) # メトリック学習の損失の計算
            loss = ce_loss + metric_loss # 総損失の計算
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_ce_loss += ce_loss.item()
        sum_metric_loss += metric_loss.item()
        sum_loss += loss.item()
        count += torch.sum(logits.argmax(dim=1) == labels).item()

    return sum_ce_loss, sum_metric_loss, sum_loss, count

def train_softmax(device, train_loader, model, optimizer, scaler, epoch, ce_loss_func, metric_head, _lambda):
    """
    学習用の関数（Softmax分類器用）

    Parameters
    ----------
    device : torch.device
        使用するデバイス（CPUまたはGPU）
    train_loader : torch.utils.data.DataLoader
        訓練データローダー
    model : nn.Module
        学習するモデル
    criterion : nn.Module
        損失関数
    optimizer : torch.optim.Optimizer
        モデルのオプティマイザ
    scaler : torch.cuda.amp.GradScaler
        自動混合精度のスケーラー
    epoch : int
        現在のエポック数
    ce_loss_func : nn.Module
        クロスエントロピー損失関数
    metric_head : nn.Module
        メトリック学習のヘッド
    _lambda : float
        メトリック学習の損失関数に対する重み
    Returns
    -------
    sum_loss : float
        訓練データに対する損失の合計
    count : int
        正しく分類されたサンプルの数
    """
    model.train()
    metric_head.train()
    
    sum_loss = 0.0
    count = 0

    for idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device, non_blocking=True).float() # 画像をデバイスに転送
        labels = labels.to(device, non_blocking=True).long() # ラベルをデバイスに転送

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, feat = model(imgs) # モデルの順伝播
            logits = metric_head(feat, labels) # メトリック学習のヘッドを通して分類結果を得る
            loss = ce_loss_func(logits, labels) # クロスエントロピー損失の計算
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logits.argmax(dim=1) == labels).item()
        
    return 0, 0, sum_loss, count

def validation(device, val_loader, model, ce_loss_func, metric_loss_func):
    """
    検証用の関数

    Parameters
    ----------
    device : torch.device
        使用するデバイス（CPUまたはGPU）
    val_loader : torch.utils.data.DataLoader
        検証データローダー
    model : nn.Module
        学習するモデル
    ce_loss_func : nn.Module
        クロスエントロピー損失関数
    metric_loss_func : nn.Module
        メトリック学習の損失関数
    Returns
    -------
    sum_loss : float
        検証データに対する損失の合計
    count : int
        正しく分類されたサンプルの数
    features_np : np.ndarray
        特徴量のNumPy配列
    labels_np : np.ndarray
        ラベルのNumPy配列
    """
    model.eval()
    
    sum_loss = 0.0
    count = 0
    
    features_list = []
    labels_list = []

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.to(device, non_blocking=True).float() # 画像をデバイスに転送
            labels = labels.to(device, non_blocking=True).long() # ラベルをデバイスに転送
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features, logits = model(imgs) # モデルの順伝播
                loss = ce_loss_func(logits, labels) # クロスエントロピー損失の計算

                features_list.append(features.cpu().detach().numpy()) # 特徴量をリストに追加
                labels_list.append(labels.cpu().detach().numpy()) # ラベルをリストに追加

            sum_loss += loss.item()
            count += torch.sum(logits.argmax(dim=1) == labels).item()

    features_np = np.concatenate(features_list, 0) # 特徴量のNumPy配列を作成
    labels_np = np.concatenate(labels_list, 0) # ラベルのNumPy配列を作成

    return sum_loss, count, features_np, labels_np

def validation_softmax(device, val_loader, model, ce_loss_func, metric_head):
    """
    検証用の関数（Softmax分類器用）

    Parameters
    ----------
    device : torch.device
        使用するデバイス（CPUまたはGPU）
    val_loader : torch.utils.data.DataLoader
        検証データローダー
    model : nn.Module
        学習するモデル
    ce_loss_func : nn.Module
        クロスエントロピー損失関数
    metric_head : nn.Module
        メトリック学習のヘッド
    Returns
    -------
    sum_loss : float
        検証データに対する損失の合計
    count : int
        正しく分類されたサンプルの数
    features_np : np.ndarray
        特徴量のNumPy配列
    labels_np : np.ndarray
        ラベルのNumPy配列
    """
    model.eval()
    metric_head.eval()
    
    sum_loss = 0.0
    count = 0
    
    features_list = []
    labels_list = []

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.to(device, non_blocking=True).float() # 画像をデバイスに転送
            labels = labels.to(device, non_blocking=True).long() # ラベルをデバイスに転送
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, features = model(imgs) # モデルの順伝播
                logits = metric_head(features, labels) # メトリック学習のヘッドを通して分類結果を得る
                loss = ce_loss_func(logits, labels) # クロスエントロピー損失の計算

                features_list.append(features.cpu().detach().numpy()) # 特徴量をリストに追加
                labels_list.append(labels.cpu().detach().numpy()) # ラベルをリストに追加

            sum_loss += loss.item()
            count += torch.sum(logits.argmax(dim=1) == labels).item()

    features_np = np.concatenate(features_list, 0) # 特徴量のNumPy配列を作成
    labels_np = np.concatenate(labels_list, 0) 

    return sum_loss, count, features_np, labels_np