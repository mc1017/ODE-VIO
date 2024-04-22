import torch
import torch.nn as nn
import torch.nn.functional as F


# The fusion module
class FusionModule(nn.Module):
    def __init__(self, feature_dim, fuse_method):
        super(FusionModule, self).__init__()
        self.fuse_method = fuse_method
        self.f_len = feature_dim
        if self.fuse_method == "soft":
            self.net = nn.Sequential(nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == "hard":
            self.net = nn.Sequential(nn.Linear(self.f_len, 2 * self.f_len))

    def forward(self, v, i):
        if self.fuse_method == "cat":
            return torch.cat((v, i), -1)
        elif self.fuse_method == "soft":
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == "hard":
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]
