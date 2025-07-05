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

    for idx, (img, label) in enumerate(tqdm(train_loader)):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logit = model(img)
                loss = criterion(logit, label)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    return sum_loss, count

def train_metric(device, train_loader, model, optimizer, scaler, epoch, ce_loss_func, metric_loss_func, _lambda):
    model.train()
    
    sum_ce_loss = 0.0
    sum_metric_loss = 0.0
    sum_loss = 0.0
    count = 0

    for idx, (img, label) in enumerate(tqdm(train_loader)):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features, logit = model(img)

            ce_loss = ce_loss_func(logit, label)
            metric_loss = _lambda * metric_loss_func(features, label)            
            loss = ce_loss + metric_loss
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_ce_loss += ce_loss.item()
        sum_metric_loss += metric_loss.item()
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    return sum_ce_loss, sum_metric_loss, sum_loss, count

def train_softmax(device, train_loader, model, optimizer, scaler, epoch, ce_loss_func, metric_head, _lambda):
    model.train()
    metric_head.train()
    
    sum_loss = 0.0
    count = 0

    for idx, (img, label) in enumerate(tqdm(train_loader)):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, feat = model(img)
            logit = metric_head(feat, label)
            loss = ce_loss_func(logit, label)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    return 0, 0, sum_loss, count

def validation(device, val_loader, model, ce_loss_func, metric_loss_func):
    model.eval()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(val_loader)):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logit = model(img)
                loss = ce_loss_func(logit, label)

            sum_loss += loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count

def validation_softmax(device, val_loader, model, ce_loss_func, metric_head):
    model.eval()
    metric_head.eval()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(val_loader)):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, feat = model(img)
                logit = metric_head(feat, label)
                loss = ce_loss_func(logit, label)

            sum_loss += loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count

def save_augmented_images(images1, images2, epoch, output_dir="augmented_samples", max_images=16):
    if epoch != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    to_pil = ToPILImage()
    images1 = images1.cpu()
    images2 = images2.cpu()

    for i in range(min(len(images1), max_images)):
        img1 = to_pil(images1[i])
        img1.save(os.path.join(output_dir, f"epoch{epoch+1}_img1{i}.png"))
        img2 = to_pil(images2[i])
        img2.save(os.path.join(output_dir, f"epoch{epoch+1}_img2{i}.png"))