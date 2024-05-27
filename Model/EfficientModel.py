import torch.nn as nn
import torch
import torchaudio.transforms as AudioT
from Model.utils import PreEmphasis
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=(1, 1),  groups=1, dilation=1):
        super(BasicBlock, self).__init__()

        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
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


        out += identity

        return out


class ResNet(nn.Module):

    def __init__(self, layers ):
        super(ResNet, self).__init__()

        self.stem = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=(1, 1), padding=1,bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True))

        self.downsampler1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3,stride=(2, 1), padding=1, bias=False),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(*[BasicBlock(planes=32,stride=(1,1)) for _ in range(layers[0])])

        self.downsampler2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3,stride=(2, 2), padding=1, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(*[BasicBlock(planes=64,stride=(1,1)) for _ in range(layers[1])])

        self.downsampler3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3,stride=(2, 1), padding=1, bias=False),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(*[BasicBlock(planes=128,stride=(1,1)) for _ in range(layers[2])])

        self.downsampler4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3,stride=(2, 1), padding=1, bias=False),
                                          nn.BatchNorm2d(256))

        self.layer4 = nn.Sequential(*[BasicBlock(planes=256,stride=(1,1)) for _ in range(layers[3])])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x):

        x = self.stem(x)

        x = self.downsampler1(x)
        x = self.layer1(x)
        x = self.downsampler2(x)

        x = self.layer2(x)

        x = self.downsampler3(x)
        x = self.layer3(x)

        x = self.downsampler4(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNet34Efficient(nn.Module):
    def __init__(self, window_length=400, hopping_length=160, mel_number=80, fft_size=256,
                 window_function=torch.hamming_window):
        super(ResNet34Efficient, self).__init__()
        self.MelSpec = nn.Sequential(PreEmphasis(),
                                     AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                           n_mels=mel_number, n_fft=fft_size,
                                                           window_fn=window_function, sample_rate=16000))
        # self.specaug = FbankAug()

        self.instancenorm = nn.InstanceNorm1d(80)
        self.model = ResNet([3, 4, 6, 3])


        self.fc = nn.Linear(in_features=2560, out_features=512)


    def forward(self, input_tensor: torch.Tensor):
        x = self.MelSpec(input_tensor) + 1e-6
        x = x.log()
        #  x = self.specaug(x)
        x = self.instancenorm(x).unsqueeze(1)
        x = self.model(x).reshape(x.shape[0], 1280,-1)

        mu = torch.mean(x, dim=-1)
        sg = torch.sqrt(torch.var(x, dim=-1) + 1e-10)
        x = torch.cat((mu, sg), 1)

        return self.fc(x)
