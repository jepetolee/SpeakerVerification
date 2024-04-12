#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch.nn as  nn
import torch
from ResnetBaseModel import resnet18
import  torchaudio.transforms as AudioT
import torch.nn.functional as F

class PreEmphasis(nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input_tensor: torch.tensor) -> torch.tensor:
        input_tensor = input_tensor.unsqueeze(1)
        input_tensor = F.pad(input_tensor, (1, 0), 'reflect')
        return F.conv1d(input_tensor, self.flipped_filter).squeeze(1)

class ResNet18_SingleChannel(nn.Module):
    def __init__(self):

            super(ResNet18_SingleChannel, self).__init__()
            self.MelSpec = nn.Sequential(PreEmphasis(),
                                         AudioT.MelSpectrogram(win_length=400, hop_length=160,
                                                               n_mels=80, n_fft=512,
                                                               window_fn=torch.hamming_window, sample_rate=16000))
            self.instancenorm = nn.InstanceNorm1d(80)
            self.model = resnet18(channel_size=1,inplane=64,embedding_size=192)

    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            return x

