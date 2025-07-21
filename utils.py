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
    """
    設定ファイルを読み込む関数

    Parameters
    ----------
    config_path : str
        設定ファイルのパス
    Returns
    -------
    config : dict
        設定内容を含む辞書
    """
    spec = importlib.util.spec_from_file_location("config_module", config_path) # 設定ファイルのモジュールを読み込む
    config_module = importlib.util.module_from_spec(spec) # モジュールを作成
    spec.loader.exec_module(config_module) # モジュールを実行

    return config_module.config

def save_model(model, optimizer, epoch, save_model_path):
    """
    モデルを保存する関数

    Parameters
    ----------
    model : nn.Module
        保存するモデル
    optimizer : torch.optim.Optimizer
        モデルのオプティマイザ
    epoch : int
        現在のエポック数
    save_model_path : str
        モデルを保存するパス
    Returns
    -------
    None
    """
    torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch
    }, save_model_path)
    print('saved model!!')

def save_log(log_path, epoch, train_loss=None, ce_loss=None, metric_loss=None, train_acc=None, val_loss=None, val_acc=None):
    """
    ログを保存する関数

    Parameters
    ----------
    log_path : str
        ログを保存するパス
    epoch : int
        現在のエポック数
    train_loss : float, optional
        訓練データに対する損失（デフォルトはNone）
    ce_loss : float, optional
        クロスエントロピー損失（デフォルトはNone）
    metric_loss : float, optional
        メトリック学習の損失（デフォルトはNone）
    train_acc : float, optional
        訓練データに対する精度（デフォルトはNone）
    val_loss : float, optional
        検証データに対する損失（デフォルトはNone）
    val_acc : float, optional
        検証データに対する精度（デフォルトはNone）
    Returns
    -------
    None
    """

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
    """
    特徴空間を可視化して保存する関数

    Parameters
    ----------
    features : np.ndarray
        特徴量の配列
    labels : np.ndarray
        ラベルの配列
    class_names : list
        クラス名のリスト
    save_featspace_path : str
        特徴空間を保存するパス
    Returns
    -------
    None
    """
    plt.figure(figsize=(10,10))
    colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"] # 色のリスト

    points = TSNE(n_components=2, random_state=0).fit_transform(features) # t-SNEで次元削減

    for i, class_name in enumerate(class_names): # クラスごとにプロット
            class_points = points[labels == i] # 特定のクラスのポイントを抽出
            plt.scatter(class_points[:, 0], class_points[:, 1], c=colors[i % len(colors)], label=class_name) # プロット

    plt.legend()
    plt.savefig(save_featspace_path)
    print('saved feature space!!')