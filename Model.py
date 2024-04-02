#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch.nn as  nn
import torch
import torch.nn.functional as F
from ResnetBaseModel import resnet18


class ResNet18_SingleChannel(nn.Module):
    def __init__(self):
            super(ResNet18_SingleChannel, self).__init__()
            self.aa = resnet18(channel_size=1,inplane=64,embedding_size=512)

    def forward(self, input_tensor: torch.Tensor):
            return self.aa(input_tensor)

