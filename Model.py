#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch.nn as  nn
import torch
import torch.nn.functional as F
from ResnetBaseModel import resnet18
import  torchaudio.transforms as AudioT


class ResNet18MFCC80(nn.Module):
    def __init__(self):
            super(ResNet18MFCC80, self).__init__()
            self.aa = resnet18(channel_size=1,inplane=64)

    def forward(self, input_tensor: torch.Tensor):
            return self.aa(input_tensor)


class ResNet18LogMel(nn.Module):
    def __init__(self):
            super(ResNet18LogMel, self).__init__()

            self.aa = resnet18(channel_size=1,inplane=64)

    def forward(self, input_tensor: torch.Tensor):
            return self.aa(input_tensor)