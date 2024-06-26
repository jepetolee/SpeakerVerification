#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as  nn
import torch
from Model.ResnetBaseModel import ResNet,BasicBlock,ResNetForEffectiveReceptiveField
import  torchaudio.transforms as AudioT
from Model.SincNet import DeformableSincConv1d,SincConv1d
from Model.utils import PreEmphasis


class ResNet44TSTP(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80,dilation=[1,1,1,1],  fft_size= 512, window_function=torch.hamming_window):

            super(ResNet44TSTP, self).__init__()
            self.MelSpec =nn.Sequential(PreEmphasis(),
                                        AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.model = ResNet(BasicBlock, [4, 4, 9, 4], in_channel=1, inplane=32,dilation=dilation)
            self.fc = nn.Linear(in_features=2560,out_features=512)

    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)+1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            pooling_mean = torch.mean(x, dim=-1)
            pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-10)
            x = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                             torch.flatten(pooling_std, start_dim=1)), 1)

            return self.fc(x)


class DeformableSincResNet44(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80,dilation=[1,1,1,1, 1,1,1],  fft_size= 512, window_function=torch.hamming_window):

            super(DeformableSincResNet44, self).__init__()
            self.Sinc =DeformableSincConv1d(out_channels=256, kernel_size=320, stride=80, padding=0)
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(256)
            self.model = ResNetForEffectiveReceptiveField(BasicBlock, [2, 2, 2, 2, 4, 6 ,3], in_channel=1, inplane=32,dilation=dilation)
            self.fc = nn.Linear(in_features=1024,out_features=512)

    def forward(self, input_tensor: torch.Tensor):
            x= self.Sinc(input_tensor) + 1e-6
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            pooling_mean = torch.mean(x, dim=-1)
            pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-10)
            x = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                             torch.flatten(pooling_std, start_dim=1)), 1)

            return self.fc(x)


class SincResNet44(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80,dilation=[1,1,1,1, 1,1,1],  fft_size= 512, window_function=torch.hamming_window):

            super(SincResNet44, self).__init__()
            self.Sinc =SincConv1d(out_channels=256, kernel_size=320, stride=80, padding=0)
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(256)
            self.model = ResNetForEffectiveReceptiveField(BasicBlock, [2, 2, 2, 2, 4, 6 ,3], in_channel=1, inplane=32,dilation=dilation)
            self.fc = nn.Linear(in_features=1024,out_features=512)

    def forward(self, input_tensor: torch.Tensor):
            x= self.Sinc(input_tensor) + 1e-6
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            pooling_mean = torch.mean(x, dim=-1)
            pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-10)
            x = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                             torch.flatten(pooling_std, start_dim=1)), 1)

            return self.fc(x)