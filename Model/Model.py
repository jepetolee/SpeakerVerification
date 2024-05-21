#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as  nn
import torch
from Model.ResnetBaseModel import resnet34Encoder
import  torchaudio.transforms as AudioT
from Model.utils import PreEmphasis,FbankAug

class ResNet34AveragePooling(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 512, window_function=torch.hamming_window):

            super(ResNet34AveragePooling, self).__init__()
            self.MelSpec = AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000)
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.model = resnet34Encoder(channel_size=1, inplane=64)
            self.fc = nn.Linear(in_features=512,out_features=512)
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
                2560, 256, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(
                256, 2560, kernel_size=1),
            nn.Softmax(dim=2), )

            if self.encoder_type == "SAP":
                self.fc = nn.Sequential(nn.Linear(in_features=2560, out_features=512),
                                        nn.GELU(),
                                        nn.BatchNorm1d(512),
                                        nn.Linear(in_features=512, out_features=256))
            elif self.encoder_type == "ASP":
                self.fc =  nn.Sequential(nn.Linear(in_features=5120, out_features=1024),
                                        nn.GELU(),
                                        nn.BatchNorm1d(1024),
                                        nn.Linear(in_features=1024, out_features=256))
    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)+1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            # B (C X F) S
            x = x.reshape((x.shape[0],-1,x.size()[-1]))

            w = self.attention(x)

            if self.encoder_type == "SAP":
                x = torch.sum(x * w, axis=2)
            elif self.encoder_type == "ASP":
                mu = torch.mean(x * w, axis=2)
                sg = torch.sum((x ** 2) * w, axis=2) - mu ** 2
                sg = torch.clip(sg, min=1e-5)
                sg = torch.sqrt(sg)
                x = torch.concat((mu, sg), 1)
            x = torch.flatten(x, 1)
            return self.fc(x)


class ResNet34SEPointwise(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 512, window_function=torch.hamming_window,encoder_type="SAP"):

            super(ResNet34SEPointwise, self).__init__()
            self.MelSpec = nn.Sequential(
                                         AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.encoder_type = encoder_type
            self.model = resnet34Encoder(channel_size=1, inplane=64)

            self.attention = nn.Sequential(
            nn.Conv2d(
                512, 256, kernel_size=(1,1)),
            nn.GELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 512, kernel_size=(1,1)),
            nn.Softmax(dim=3))


            self.fc = nn.Linear(in_features=2560, out_features=256)
    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)+1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)

            w = self.attention(x)

            x = torch.sum(x * w, axis=-1)

            x = torch.flatten(x, 1)
            return  self.fc(x)


class ResNet34DoubleAttention(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 512, window_function=torch.hamming_window,encoder_type="SAP"):

            super(ResNet34DoubleAttention, self).__init__()
            self.MelSpec = nn.Sequential(PreEmphasis(),
                                         AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.encoder_type = encoder_type
            self.model = resnet34Encoder(channel_size=1, inplane=64)

            self.attention = nn.Sequential(
            nn.Conv2d(
                512, 256, kernel_size=(1,1)),
            nn.GELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 512, kernel_size=(1,1)),
            nn.Softmax(dim=3))

            self.attention2 = nn.Sequential(
                nn.Conv1d(
                    512, 128, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(128),
                nn.Conv1d(
                    128, 512, kernel_size=1),
                nn.Softmax(dim=2))

            self.FeedGForwardlayer2 = nn.Sequential(nn.Linear(5, 1),
                                             nn.BatchNorm1d(512),
                                             nn.GELU())

            self.fc = nn.Linear(in_features=512, out_features=256)
    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)+1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)

            w = self.attention(x)

            x = torch.sum(x * w, axis=-1)

            x = self.attention2(x)

            x= self.FeedGForwardlayer2(x)

            x = torch.flatten(x, 1)
            return self.fc(x)




