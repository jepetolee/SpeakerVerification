#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple
from sklearn import metrics

# explain: build EER from scores and labels
def computeEER(scores:list, labels:list)->float:

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100
    return eer

# Adapted from https://github.com/wujiyang/Face_Pytorch
class AAM_Softmax(nn.Module):
    def __init__(self, n_class:int, margin:float, scale:int):
        super(AAM_Softmax, self).__init__()

        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 512), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.margin = margin
        self.scale:int = scale
    
        self.AngularLoss = nn.CrossEntropyLoss()
       
        self.margined_cosine = math.cos(self.margin)
        self.margined_sine = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, x:torch.Tensor, label:torch.Tensor=None)-> Tuple[torch.Tensor, torch.Tensor, tuple]:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))

        phi = cosine * self.margined_cosine - sine * self.margined_sine
        phi = torch.where((cosine - self.threshold) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = self.AngularLoss(output, label)

        return loss