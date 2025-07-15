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
    model.train()
    
    sum_loss = 0.0
    count = 0

    for idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logits = model(imgs)
                loss = criterion(logits, labels)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logits.argmax(dim=1) == labels).item()
        
    return sum_loss, count

def train_metric(device, train_loader, model, optimizer, scaler, epoch, ce_loss_func, metric_loss_func, _lambda):
    model.train()
    
    sum_ce_loss = 0.0
    sum_metric_loss = 0.0
    sum_loss = 0.0
    count = 0

    for idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features, logits = model(imgs)
            ce_loss = ce_loss_func(logits, labels)
            metric_loss = _lambda * metric_loss_func(features, labels)            
            loss = ce_loss + metric_loss
            
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
    model.train()
    metric_head.train()
    
    sum_loss = 0.0
    count = 0

    for idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, feat = model(imgs)
            logits = metric_head(feat, labels)
            loss = ce_loss_func(logits, labels)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logits.argmax(dim=1) == labels).item()
        
    return 0, 0, sum_loss, count

def validation(device, val_loader, model, ce_loss_func, metric_loss_func):
    model.eval()
    
    sum_loss = 0.0
    count = 0
    
    features_list = []
    labels_list = []

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features, logits = model(imgs)
                loss = ce_loss_func(logits, labels)

                features_list.append(features.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())

            sum_loss += loss.item()
            count += torch.sum(logits.argmax(dim=1) == labels).item()

    features_np = np.concatenate(features_list, 0)
    labels_np = np.concatenate(labels_list, 0)

    return sum_loss, count, features_np, labels_np

def validation_softmax(device, val_loader, model, ce_loss_func, metric_head):
    model.eval()
    metric_head.eval()
    
    sum_loss = 0.0
    count = 0
    
    features_list = []
    labels_list = []

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, features = model(imgs)
                logits = metric_head(features, labels)
                loss = ce_loss_func(logits, labels)

                features_list.append(features.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())

            sum_loss += loss.item()
            count += torch.sum(logits.argmax(dim=1) == labels).item()

    features_np = np.concatenate(features_list, 0)
    labels_np = np.concatenate(labels_list, 0)

    return sum_loss, count, features_np, labels_np