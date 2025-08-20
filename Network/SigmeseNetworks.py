import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Global Average Pooling
class CenterLoss(nn.Module):
    def __init__(self, num_classes=5, feat_dim=256., device='cuda', alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.alpha = alpha  # 中心更新步长
        self.centers = nn.Parameter(torch.randn(size=[int(num_classes), int(feat_dim)]).to(device), requires_grad=True)  # 不自动反传

    def forward(self, features, labels):
        """
        features: tensor of shape (batch_size, 384, 32, 11)
        labels: tensor of shape (batch_size,)
        """
        batch_size = features.size(0)
        d_features = features.reshape(batch_size, -1)

        centers_batch = self.centers[labels-1]  # (batch_size, feat_dim)
        loss = (d_features - centers_batch).pow(2).sum(dim=1).mean()

        # 更新中心
        # self._update_centers(d_features, labels-1)

        return loss

    @torch.no_grad()
    def _update_centers(self, features, labels):
        """
        更新类别中心，features 已经是 (batch_size, feat_dim)
        """
        # 统计每个类别出现的次数
        unique_labels, counts = labels.unique(return_counts=True)

        for label in unique_labels:
            mask = (labels == label)
            features_of_label = features[mask]  # 选出当前label对应的特征
            mean_feature = features_of_label.mean(dim=0)

            # 更新中心
            self.centers[label] -= self.alpha * (self.centers[label] - mean_feature)

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features, labels):
        """
        features: tensor of shape (batch_size, 384, 32, 11)
        labels: tensor of shape (batch_size,)
        """
        batch_size = features.size(0)
        d_features = features.reshape(batch_size, -1)

        n = d_features.size(0)

        # 计算 pairwise distance
        dist = torch.cdist(d_features, d_features, p=2)  # shape: (batch_size, batch_size)

        # 选正负样本
        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())  # 正样本mask
        mask_neg = labels.expand(n, n).ne(labels.expand(n, n).t())  # 负样本mask

        # 每个anchor找 hardest positive 和 hardest negative
        dist_ap = torch.max(dist * mask_pos.cuda().float(), dim=1)[0]  # hardest positive
        dist_an = torch.min(dist + (1.0 - mask_neg.cuda().float()) * 1e6, dim=1)[0]  # hardest negative

        # 计算 triplet loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
