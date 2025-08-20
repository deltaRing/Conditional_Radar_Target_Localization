import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def mse_loss(pred, ground_truth):
    mse = F.mse_loss(pred, ground_truth)
    return mse


def kl_loss(pred, target):
    pred_log = F.log_softmax(pred, dim=-1)
    q        = F.softmax(target, dim=-1)
    kl = F.kl_div(pred_log, q, reduction='batchmean')
    return kl

def cls_loss(pred, ground_truth):
    ce_loss = torch.mean(-(ground_truth * torch.log(pred + 1e-8) + (1 - ground_truth + 1e-8) * torch.log(1 - pred + 1e-8)))
    return ce_loss

def clsfi_loss(pred, ground_truth):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, ground_truth)
    return loss
