import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # 特徴量間のペアワイズ距離の計算
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() # 同じラベルのマスク（ポジティブ）
        negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float() # 異なるラベルのマスク（ネガティブ）

        positive_dist = pairwise_dist * positive_mask
        negative_dist = pairwise_dist * negative_mask

        # 各ペアの数をカウント
        num_pos_pairs = positive_mask.sum().item()
        num_neg_pairs = negative_mask.sum().item()

        if num_pos_pairs == 0 or num_neg_pairs == 0:
            return torch.tensor(0.0)

        # Contrastive Loss
        loss = (
            (positive_dist.pow(2) / 2).mean() +  
            (F.relu(self.margin - negative_dist + 1e-9).pow(2) / 2).mean() 
        )

        return loss

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, hard_triplets=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.use_hard_triplets = hard_triplets

    def forward(self, embeddings, labels):
        # 特徴量間のペアワイズ距離の計算
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() # 同じラベルのマスク（ポジティブ）
        negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float() # 異なるラベルのマスク（ネガティブ）
        # print('Pos Mask:', positive_mask)
        # print('Neg Mask:', negative_mask)

        if self.use_hard_triplets: # Hard PositiveおよびHard Negativeを選択
            positive_dist = pairwise_dist * positive_mask
            positive_dist = positive_dist + (1 - positive_mask) * -1e6
            hardest_positive_dist, _ = positive_dist.max(dim=1) # 各アンカーに対する最も遠いポジティブ

            negative_dist = pairwise_dist + (1 - negative_mask) * 1e6 # 無効なネガティブに大きな値を設定
            hardest_negative_dist, _ = negative_dist.min(dim=1) # 各アンカーに対する最も近いネガティブ

            # Triplet Loss
            loss = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            return loss.mean()

        else: # 全ペアを考慮
            positive_dist = pairwise_dist * positive_mask
            # print('Pos Dist:', positive_dist)
            negative_dist = pairwise_dist * negative_mask
            # print('Neg Dist:', negative_dist)

            # Triplet Loss
            triplet_loss = positive_dist.unsqueeze(2) - negative_dist.unsqueeze(1) + self.margin 
            triplet_loss = torch.relu(triplet_loss) # マージンに基づくReLU適用

            valid_triplets = positive_mask.unsqueeze(2) * negative_mask.unsqueeze(1) # 有効なTripletのマスク
            triplet_loss = triplet_loss * valid_triplets
            num_valid_triplets = valid_triplets.sum() + 1e-16 # 有効ペア数（ゼロ除算を防ぐ）
            loss = triplet_loss.sum() / num_valid_triplets
            return loss
        
# ArcFace
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

# CosFace
class CosFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(CosFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

# SphereFace
class SphereFaceHead(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(SphereFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'