#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(inplanes)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # https://arxiv.org/pdf/2302.06112.pdf
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.gelu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.gelu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=80, inplane=128, zero_init_residual=False,  groups=1, width_per_group=64,dilation=[1,1,1,1],norm_layer= nn.BatchNorm2d):
        super(ResNet, self).__init__()

        self._norm_layer = norm_layer
        self.inplanes = inplane
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn = norm_layer(256)
        self.gelu = nn.GELU()

        self.layer1 = self._make_layer(block, 32, layers[0], stride=(2,1), dilate=dilation[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=(2,2), dilate=dilation[1])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=(2,1), dilate=dilation[2])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=(2,1), dilate=dilation[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,dilation= dilate),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, dilate, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, dilation=dilate, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn(x)
        x = self.gelu(x)
        return x


class ResNetForEffectiveReceptiveField(nn.Module):

    def __init__(self, block, layers, in_channel=80, inplane=128, zero_init_residual=False,  groups=1, width_per_group=64,dilation=[1,1,1,1],norm_layer= nn.BatchNorm2d):
        super(ResNetForEffectiveReceptiveField, self).__init__()

        self._norm_layer = norm_layer
        self.inplanes = inplane
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn = norm_layer(256)
        self.gelu = nn.GELU()

        self.layer1 = self._make_layer(block, 4, layers[0], stride=(2,1), dilate=dilation[0])
        self.layer2 = self._make_layer(block, 8, layers[1], stride=(2,1), dilate=dilation[1])
        self.layer3 = self._make_layer(block, 16, layers[2], stride=(2,1), dilate=dilation[2])
        self.layer4 = self._make_layer(block, 32, layers[3], stride=(2,1), dilate=dilation[3])
        self.layer5 = self._make_layer(block, 64, layers[4], stride=(2,2), dilate=dilation[4])
        self.layer6 = self._make_layer(block, 128, layers[5], stride=(2,1), dilate=dilation[5])
        self.layer7 = self._make_layer(block, 256, layers[6], stride=(2,1), dilate=dilation[6])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,dilation= dilate),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, dilate, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, dilation=dilate, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
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