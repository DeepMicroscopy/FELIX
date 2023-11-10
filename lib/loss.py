"""

Adapted from https://github.com/hongxin001/logitnorm_ood/blob/main/common/loss_function.py

"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class LogitNormLoss(nn.Module):

    def __init__(self, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, input, target):
        norms = torch.norm(input, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(input, norms) / self.t
        return F.binary_cross_entropy_with_logits(logit_norm, target)
    
    