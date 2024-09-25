import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from timm.loss import SoftTargetCrossEntropy
import torchvision
# from torch.optim.lr_scheduler import _LRScheduler
import math
from torch import nn

# class CustomCosineAnnealingLR(_LRScheduler):
#     def __init__(self, optimizer, half_epochs, min_lr, last_epoch=-1):
#         self.half_epochs = half_epochs
#         self.min_lr = min_lr
#         super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         if self.last_epoch >= self.half_epochs:
#             return [self.min_lr] * len(self.base_lrs)
#         else:
#             return [base_lr + (self.min_lr - base_lr) * (1 + math.cos(math.pi * self.last_epoch / self.half_epochs)) / 2
#                     for base_lr in self.base_lrs]


def imshow(img):
    img = torchvision.utils.make_grid(img)
    torchvision.utils.save_image(img,"111.png")

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def train_one_epoch(model, optimizer, data_loader, device, epoch, mixup_fn,is_warmup=True):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=100)
    lr_scheduler = None
    if epoch == 0 and is_warmup:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(10000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        # s = torch.sum(model.leaky_att_network[0].leaky_attention.conv.weight.data).item()
        # ss = torch.sum(model.leaky_att_network[0].leaky_attention.trans.attn.qkv.weight.data).item()
        # sss = torch.sum(model.first_conv.layer1[0].weight.data).item()
        # data_loader.desc = "loss: {:.5f}".format(accu_loss.item() / (step + 1))+\
        #                         f"mask_conv:{s:.3f},qkv{ss:.3f},first_conv{sss:.3f}"
        data_loader.desc = "[t epoch {}] loss: {:.3f}".format(epoch,accu_loss.item() / (step + 1)) + f" lr:{format(optimizer.param_groups[0]['lr'],'.1E')}"
                                                                               # +accu_num.item() / sample_num,acc: {:.3f}

        if not torch.isfinite(loss):  # 检查是不是爆炸了 (infinite)
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

    return accu_loss.item() / (step + 1) ,0#, accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=100)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
