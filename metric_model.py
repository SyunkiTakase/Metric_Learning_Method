import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet152, vgg11, vgg13, vgg16, vgg19

import base_model

class MetricModel(nn.Module):
    """
    メトリック学習モデルの定義

    Parameters
    ----------
    method : str
        使用するメトリック学習手法の名前
    arch : str
        使用するアーキテクチャの名前
    num_dim : int
        特徴量の次元数
    num_classes : int
        分類クラス数
    Returns
    -------
    None
    """
    def __init__(self, method='SiameseNetwork', arch='ResNet18', num_dim=512, num_classes=10):
        super(MetricModel, self).__init__()
        self.method = method
        self.arch = arch
        self.num_dim = num_dim
        self.num_classes = num_classes
        
        self.selected_arch(arch=self.arch)
        last_layers = ['fc', 'classifier', 'heads'] # 最終層の名前リスト

        for layer in last_layers: # 最終層の名前を順に確認
            if hasattr(self.encoder, layer):
                # 最終層がSequentialの場合
                if isinstance(getattr(self.encoder, layer), nn.Sequential):
                    self.out_enc_dim = getattr(self.encoder, layer)[0].in_features
                else:
                    self.out_enc_dim = getattr(self.encoder, layer).in_features
                break

        for layer in last_layers: # 最終層の名前を順に確認
            if hasattr(self.encoder, layer):
                # 最終層がSequentialの場合
                if isinstance(getattr(self.encoder, layer), nn.Sequential):
                    setattr(self.encoder, layer, nn.Sequential(nn.Identity()))
                else:
                    setattr(self.encoder, layer, nn.Identity())
                break

        if self.method == 'ArcFace' or self.method == 'CosFace' or self.method == 'SphereFace': # ArcFace/CosFace/SphereFaceを使用する場合
            self.classifier = nn.Linear(self.out_enc_dim, self.num_dim) # 特徴量の次元をnum_dimに変更
        else: # その他の手法を使用する場合
            self.classifier = nn.Linear(self.out_enc_dim, self.num_classes) # 分類クラス数に変更

    def forward(self, x):
        """
        順伝播の定義

        Parameters
        ----------
        x : torch.Tensor
            入力画像のテンソル
        Returns
        -------
        feat : torch.Tensor
            特徴量
        y : torch.Tensor
            分類結果
        """
        feat = self.encoder(x)
        y = self.classifier(feat)

        return feat, y
    
    def selected_arch(self, arch):
        """
        使用するアーキテクチャを選択する

        Parameters
        ----------
        arch : str
            使用するアーキテクチャの名前
        Returns
        -------
        None
        """
        arch_map = {
            'ResNet18': resnet18,
            'ResNet34': resnet34,
            'ResNet50': resnet50,
            'ResNet152': resnet152,
            'VGG11': vgg11,
            'VGG13': vgg13,
            'VGG16': vgg16,
            'VGG19': vgg19,
        }
        
        if arch in arch_map:
            self.encoder = arch_map[arch](weights='IMAGENET1K_V1')

        else:
            module_name = f'base_model.{arch}'
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                raise ValueError(f'Unsupported architecture: {arch} (module {module_name} not found)') from e

            try:
                model_cls = getattr(module, arch)
            except AttributeError as e:
                raise ValueError(f'Module {module_name} does not define class {arch}') from e

            self.encoder = model_cls()