import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet152, vgg11, vgg13, vgg16, vgg19

class MetricModel(nn.Module):
    def __init__(self, method='SiameseNetwork', arch='ResNet18', num_dim=512, num_classes=10):
        super(MetricModel, self).__init__()
        self.method = method
        self.arch = arch
        self.num_dim = num_dim
        self.num_classes = num_classes
        
        self.selected_arch(arch=self.arch)
        last_layers = ['fc', 'classifier', 'heads']

        for layer in last_layers:
            if hasattr(self.encoder, layer):
                # 最終層がSequentialの場合
                if isinstance(getattr(self.encoder, layer), nn.Sequential):
                    self.out_enc_dim = getattr(self.encoder, layer)[0].in_features
                else:
                    self.out_enc_dim = getattr(self.encoder, layer).in_features
                break

        for layer in last_layers:
            if hasattr(self.encoder, layer):
                # 最終層がSequentialの場合
                if isinstance(getattr(self.encoder, layer), nn.Sequential):
                    setattr(self.encoder, layer, nn.Sequential(nn.Identity()))
                else:
                    setattr(self.encoder, layer, nn.Identity())
                break

        if self.method == 'ArcFace' or self.method == 'CosFace' or self.method == 'SphereFace':
            self.classifier = nn.Linear(self.out_enc_dim, self.num_dim)
        else:
            self.classifier = nn.Linear(self.out_enc_dim, self.num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        y = self.classifier(feat)

        return feat, y
    
    def selected_arch(self, arch):
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
        
        if arch not in arch_map:
            raise ValueError(f'Unsupported architecture: {arch}')
        
        self.encoder = arch_map[arch](weights='IMAGENET1K_V1')