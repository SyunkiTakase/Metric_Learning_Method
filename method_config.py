import torch.optim as optim

from metric_model import MetricModel
from trainer import train_metric, train_softmax, validation, validation_softmax
from metric_loss import ContrastiveLoss, TripletLoss, ArcFaceHead, CosFaceHead, SphereFaceHead

def build_metric_method(method, arch, lr, num_classes , num_dim, margin, scale, hard_triplets, easy_margin, device):
    if method == 'SiameseNetwork':
        model = MetricModel(method=method, arch=arch, num_classes=num_classes).to(device)
        metric_loss_func = ContrastiveLoss(margin=margin).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        return model, metric_loss_func, optimizer, train_metric, validation

    elif method == 'TripletLoss':
        model = MetricModel(method=method, arch=arch, num_classes=num_classes).to(device)
        metric_loss_func = TripletLoss(margin=margin, hard_triplets=hard_triplets).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        return model, metric_loss_func, optimizer, train_metric, validation

    elif method == 'ArcFace':
        model = MetricModel(method=method, arch=arch, num_dim=num_dim, num_classes=num_classes).to(device)
        metric_loss_func = ArcFaceHead(num_dim, num_classes, s=scale, m=margin, easy_margin=easy_margin).to(device)
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_loss_func.parameters()}],
                            lr=lr, 
                            weight_decay=5e-4)

        return model, metric_loss_func, optimizer, train_softmax, validation_softmax

    elif method == 'CosFace':
        model = MetricModel(method=method, arch=arch, num_dim=num_dim, num_classes=num_classes).to(device)
        metric_loss_func = CosFaceHead(num_dim, num_classes, s=scale, m=margin).to(device)
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_loss_func.parameters()}],
                            lr=lr, 
                            weight_decay=5e-4)
        
        return model, metric_loss_func, optimizer, train_softmax, validation_softmax

    elif method == 'SphereFace':
        model = MetricModel(method=method, arch=arch, num_dim=num_dim, num_classes=num_classes).to(device)
        metric_loss_func = SphereFaceHead(num_dim, num_classes, m=margin).to(device)
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_loss_func.parameters()}],
                            lr=lr, 
                            weight_decay=5e-4)

        return model, metric_loss_func, optimizer, train_softmax, validation_softmax

    else:
        raise ValueError(f"Unsupported method: {method}")