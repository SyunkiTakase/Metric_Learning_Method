import torch.optim as optim

from metric_model import MetricModel
from trainer import train_metric, train_softmax, validation, validation_softmax
from metric_loss import ContrastiveLoss, TripletLoss, ArcFaceHead, CosFaceHead, SphereFaceHead

def build_metric_method(method, arch, lr, weight_decay, optim_momentum, num_classes , num_dim, margin, scale, hard_triplets, easy_margin, device):
    """
    Metric Learning Methodを構築する関数

    Parameters
    ----------
    method : str
        使用するメトリック学習手法の名前
    arch : str
        使用するアーキテクチャの名前
    lr : float
        学習率
    weight_decay : float
        重み減衰
    optim_momentum : float
        オプティマイザのモメンタム
    num_classes : int
        クラス数
    num_dim : int
        特徴量の次元数
    margin : float
        マージンの値
    scale : float
        スケーリングの値
    hard_triplets : bool
        ハードトリプレットを使用するかどうか
    easy_margin : bool
        イージーマージンを使用するかどうか
    device : torch.device
        使用するデバイス（CPUまたはGPU）
    Returns
    -------
    model : MetricModel
        構築されたメトリック学習モデル
    metric_loss_func : nn.Module
        メトリック学習の損失関数
    optimizer : torch.optim.Optimizer
        モデルのオプティマイザ 
    train_fn : function
        学習用の関数
    validation_fn : function
        検証用の関数
    """
    if method == 'SiameseNetwork': # Siamese Networkを使用する場合
        model = MetricModel(method=method, arch=arch, num_classes=num_classes).to(device) # モデルの初期化
        metric_loss_func = ContrastiveLoss(margin=margin).to(device) # 損失関数の初期化
        optimizer = optim.Adam(model.parameters(), lr=lr) # オプティマイザの初期化

        return model, metric_loss_func, optimizer, train_metric, validation

    elif method == 'TripletLoss': # Triplet Lossを使用する場合
        model = MetricModel(method=method, arch=arch, num_classes=num_classes).to(device) # モデルの初期化
        metric_loss_func = TripletLoss(margin=margin, hard_triplets=hard_triplets).to(device) # 損失関数の初期化
        optimizer = optim.SGD(model.parameters(), # 学習率とモメンタムを指定
                            lr=lr, 
                            momentum=optim_momentum,
                            weight_decay=weight_decay)
        return model, metric_loss_func, optimizer, train_metric, validation

    elif method == 'ArcFace': # ArcFaceを使用する場合
        model = MetricModel(method=method, arch=arch, num_dim=num_dim, num_classes=num_classes).to(device) # モデルの初期化
        metric_loss_func = ArcFaceHead(num_dim, num_classes, s=scale, m=margin, easy_margin=easy_margin).to(device) # 損失関数の初期化
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_loss_func.parameters()}], # 学習率とモメンタムを指定
                            lr=lr, 
                            momentum=optim_momentum,
                            weight_decay=weight_decay)

        return model, metric_loss_func, optimizer, train_softmax, validation_softmax

    elif method == 'CosFace': # CosFaceを使用する場合
        model = MetricModel(method=method, arch=arch, num_dim=num_dim, num_classes=num_classes).to(device) # モデルの初期化
        metric_loss_func = CosFaceHead(num_dim, num_classes, s=scale, m=margin).to(device) # 損失関数の初期化
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_loss_func.parameters()}], # 学習率とモメンタムを指定
                            lr=lr, 
                            momentum=optim_momentum,
                            weight_decay=weight_decay)
        
        return model, metric_loss_func, optimizer, train_softmax, validation_softmax

    elif method == 'SphereFace': # SphereFaceを使用する場合
        model = MetricModel(method=method, arch=arch, num_dim=num_dim, num_classes=num_classes).to(device) # モデルの初期化
        metric_loss_func = SphereFaceHead(num_dim, num_classes, m=margin).to(device) # 損失関数の初期化
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_loss_func.parameters()}], # 学習率とモメンタムを指定
                            lr=lr, 
                            momentum=optim_momentum,
                            weight_decay=weight_decay)

        return model, metric_loss_func, optimizer, train_softmax, validation_softmax

    else:
        raise ValueError(f"Unsupported method: {method}")