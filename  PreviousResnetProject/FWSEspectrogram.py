import torch.nn as nn
import torch
import  torchaudio.transforms as AudioT
from Model.Model import PreEmphasis


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute(0,2,1,3)
        w = self.avg_pool(x).reshape((b, f))
        w = self.fc(w).reshape((b, f, 1, 1))
        x = x*w
        return x.permute(0,2,1,3)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,stride=(1, 1), downsample=None, dilation=1, norm_layer=None,frequency=40 ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation is not False:
            self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.se = SELayer(frequency, 2)
        self.stride = stride

    # https://arxiv.org/pdf/2302.06112.pdf
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=(1,1), padding=1)
        self.bn = norm_layer(8)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 8, layers[0],stride=(2,1),frequency=129)
        self.layer2 = self._make_layer(block, 16, layers[1], stride=(2,1),frequency=65,
                                       dilate=1)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=(2,2),frequency=33,
                                       dilate=1)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=(2,1),frequency=17,
                                       dilate=1)
        self.layer5 = self._make_layer(block, 128, layers[4], stride=(2,1),frequency=9,
                                       dilate=1)
        self.layer6 = self._make_layer(block, 256, layers[5], stride=(2,1),frequency=5,
                                       dilate=1)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,frequency=40):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, previous_dilation, norm_layer,frequency))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilate,
                                norm_layer=norm_layer,frequency=frequency))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet34Encoder(channel_size=80,  **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 4, 3 ,3], in_channel=channel_size,  **kwargs)

class ResNet34FWSESpectrogram(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 256, window_function=torch.hamming_window):

            super(ResNet34FWSESpectrogram, self).__init__()

            self.Spec =  nn.Sequential(AudioT.Spectrogram(win_length=window_length, hop_length=hopping_length,
                                                                n_fft=fft_size,
                                                               window_fn=window_function))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(257)
            self.model = resnet34Encoder(channel_size=1)
            self.fc = nn.Linear(in_features=2560,out_features=512)

    def forward(self, input_tensor: torch.Tensor):
            x= self.Spec(input_tensor) + 1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            pooling_mean = torch.mean(x, dim=-1)
            pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-10)
            x = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                             torch.flatten(pooling_std, start_dim=1)), 1)

            return self.fc(x)
