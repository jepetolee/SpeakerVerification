#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch.nn as  nn
import torch
from Models.ResnetBaseModel import resnet34Encoder
import  torchaudio.transforms as AudioT
import torch.nn.functional as F
from Models.utils import PreEmphasis,FbankAug




class ResNet34AveragePooling(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 512, window_function=torch.hamming_window):

            super(ResNet34AveragePooling, self).__init__()
            self.MelSpec = nn.Sequential(PreEmphasis(),
                                         AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.model = resnet34Encoder(channel_size=1, inplane=64)
            self.fc = nn.Linear(in_features=512,out_features=256)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)+1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            return self.fc(x)


class ResNet34SE(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 512, window_function=torch.hamming_window,encoder_type="SAP"):

            super(ResNet34SE, self).__init__()
            self.MelSpec = nn.Sequential(PreEmphasis(),
                                         AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.encoder_type = encoder_type
            self.model = resnet34Encoder(channel_size=1, inplane=64)



            self.attention = nn.Sequential(
            nn.Conv1d(
                512, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(
                128, 512, kernel_size=1),
            nn.Softmax(dim=2), )

            if self.encoder_type == "SAP":
                self.fc = nn.Linear(in_features=512, out_features=256)
            elif self.encoder_type == "ASP":
                self.fc = nn.Linear(in_features=1024, out_features=256)
    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)+1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            x = x.reshape((x.shape[0],512,-1))
            w = self.attention(x)

            if self.encoder_type == "SAP":
                x = torch.sum(x * w, axis=2)
            elif self.encoder_type == "ASP":
                mu = torch.sum(x * w, axis=2)
                sg = torch.sum((x ** 2) * w, axis=2) - mu ** 2
                sg = torch.clip(sg, min=1e-5)
                sg = torch.sqrt(sg)
                x = torch.concat((mu, sg), 1)
            x = torch.flatten(x, 1)
            return self.fc(x)