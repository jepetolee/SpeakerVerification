#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch.nn as  nn
import torch
import torch.nn.functional as F
from ResnetBaseModel import resnet18

class TestingModel(nn.Module):
    def __init__(self):
            super(TestingModel, self).__init__()
            self.aa = resnet18()
            self.FC = nn.Sequential(nn.Linear(1000,192),
                                    nn.BatchNorm1d(192))

    def forward(self, input_tensor: torch.Tensor):
            x = self.aa(input_tensor)
            x = F.gelu(x)
            return self.FC(x)