#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import torch
import torchaudio.transforms as AudioT
from Model.utils import PreEmphasis
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # https://arxiv.org/pdf/2302.06112.pdf
    def forward(self, x):
        identity = x
        # P1
        out = self.bn1(x)

        # P2
        out = self.gelu(out)

        # P3
        out = self.conv1(out)

        # P4
        out = self.bn2(out)

        out = self.gelu(out)
        # P6
        out = self.conv2(out)
        # P7

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=80,inplane=128, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn = norm_layer(512)
        self.gelu = nn.GELU()
        self.layer1 = self._make_layer(block, 8, layers[0],stride=(2,1))
        self.layer2 = self._make_layer(block, 16, layers[1], stride=(2,1),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 32, layers[2], stride=(2,1),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 64, layers[3], stride=(2,2),
                                       dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_layer(block, 128, layers[4],stride=(2,1))
        self.layer6 = self._make_layer(block, 256, layers[5], stride=(2,1),
                                       dilate=replace_stride_with_dilation[0])
        self.layer7 = self._make_layer(block, 512, layers[6], stride=(2, 1),
                                       dilate=replace_stride_with_dilation[0])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.bn(x)
        x = self.gelu(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet34Encoder(channel_size=1,  **kwargs):
    return ResNet(BasicBlock, [1, 1, 2, 2, 5 ,3, 2], in_channel=channel_size,  **kwargs)


class ResNet34Spectrogram(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 512, window_function=torch.hamming_window):

            super(ResNet34Spectrogram, self).__init__()
            self.Spectrogram =nn.Sequential(
                                        AudioT.Spectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_fft=fft_size,
                                                               window_fn=window_function))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(257)
            self.model = resnet34Encoder(channel_size=1)
            self.fc = nn.Linear(in_features=3072,out_features=512)

    def forward(self, input_tensor: torch.Tensor):
            x= self.Spectrogram(input_tensor) + 1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            pooling_mean = torch.mean(x, dim=-1)
            pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-10)
            x = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                             torch.flatten(pooling_std, start_dim=1)), 1)

            return self.fc(x)