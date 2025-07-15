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

def load_config(config_path):

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.config

def save_model(model, optimizer, epoch, save_model_path):

    torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch
    }, save_model_path)
    print('saved model!!')

def save_log(log_path, epoch, train_loss=None, ce_loss=None, metric_loss=None, train_acc=None, val_loss=None, val_acc=None):

    new_row = [
        epoch,
        train_loss if train_loss is not None else '',
        ce_loss if ce_loss is not None else '',
        metric_loss if metric_loss is not None else '',
        train_acc if train_acc is not None else '',
        val_loss if val_loss is not None else '',
        val_acc if val_acc is not None else ''
    ]
    
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)
    print('saved log!!')

def save_featspace(features, labels, class_names, save_featspace_path):

    plt.figure(figsize=(10,10))
    colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]

    points = TSNE(n_components=2, random_state=0).fit_transform(features)

    for i, class_name in enumerate(class_names):
            class_points = points[labels == i]
            plt.scatter(class_points[:, 0], class_points[:, 1], c=colors[i % len(colors)], label=class_name)

    plt.legend()
    plt.savefig(save_featspace_path)
    print('saved feature space!!')