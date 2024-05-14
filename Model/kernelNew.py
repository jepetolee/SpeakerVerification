import torch.nn as nn
import torch
from Model.ResnetBaseModel import resnet34Encoder
import  torchaudio.transforms as AudioT
from Model.utils import PreEmphasis,FbankAug

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None, groups=1,
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
        self.bn1 = norm_layer(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # https://arxiv.org/pdf/2302.06112.pdf
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gelu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.gelu(out)
        out = self.bn2(out)

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

        self.inplanes = inplane
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
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=(2,1), padding=0,
                               bias=False)
        self.bn = norm_layer(64)
        self.gelu = nn.GELU()
        self.layer1 = self._make_layer(block, 64, layers[0],stride=(2,1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2,1),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2,1),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2,1),
                                       dilate=replace_stride_with_dilation[2])

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

        x = self.conv1(x)
        x = self.gelu(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet34Encoder(channel_size=80, inplane=64, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channel=channel_size, inplane=inplane, **kwargs)

class ResNet34KernelExaggerate(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 256, window_function=torch.hamming_window):

            super(ResNet34KernelExaggerate, self).__init__()
            self.MelSpec = nn.Sequential(PreEmphasis(),                    
                                         AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000))
            #self.specaug = FbankAug()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.model = resnet34Encoder(channel_size=1, inplane=64)
            self.conv1d_reader = nn.Sequential(nn.Conv1d(in_channels=1536, out_channels=512,kernel_size=1),
                                               nn.GELU(),
                                               nn.BatchNorm1d(512))
            self.fc = nn.Linear(in_features=1024,out_features=512)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, input_tensor: torch.Tensor):
            x= self.MelSpec(input_tensor)+1e-6
            x = x.log()
          #  x = self.specaug(x)
            x = self.instancenorm(x).unsqueeze(1)
            x = self.model(x)
            x = x.reshape(-1,512*3,x.shape[3])
            x = self.conv1d_reader(x)
            mean = torch.mean(x, dim=2)
            deviations = x - mean.unsqueeze(2)
            variance = torch.mean(deviations ** 2, dim=2)

            pooled_features = torch.cat((mean, variance), dim=1)

            return self.fc(pooled_features)
