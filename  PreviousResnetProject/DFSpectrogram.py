import torch.nn as nn
import torch
import torchaudio.transforms as AudioT
from Model.utils import PreEmphasis
import torch.nn.functional as F


class InvertBNFBlock(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(InvertBNFBlock, self).__init__()


        self.conv1 = nn.Sequential(nn.Conv2d(planes, planes*4, kernel_size=1,bias=False),
                    nn.BatchNorm2d(planes*4))

        self.conv2 = nn.Sequential(nn.Conv2d(planes*4, planes*4,kernel_size=3, padding=1, groups=4 * planes, bias=False),
                    nn.BatchNorm2d(planes*4))

        self.conv3 = nn.Sequential(nn.Conv2d(planes*4, planes, kernel_size=1, bias=False),
                    nn.BatchNorm2d(planes))

    # https://arxiv.org/pdf/2302.06112.pdf
    def forward(self, x):
        out = F.gelu(self.conv1(x))
        out = F.gelu(self.conv2(out))
        out = self.conv3(out)
        out += x
        out = F.gelu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, layers ):
        super(ResNet, self).__init__()

        self.stem = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=(1, 1), padding=1,bias=False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(inplace=True))

        self.downsampler1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3,stride=(2, 1), padding=1, bias=False),
                                          nn.BatchNorm2d(32))
        self.layer1 = nn.Sequential(*[InvertBNFBlock(planes=32) for _ in range(layers[0])])

        self.downsampler2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3,stride=(2, 2), padding=1, bias=False),
                                          nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(*[InvertBNFBlock(planes=64) for _ in range(layers[1])])

        self.downsampler3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3,stride=(2, 1), padding=1, bias=False),
                                          nn.BatchNorm2d(128))
        self.layer3 = nn.Sequential(*[InvertBNFBlock(planes=128) for _ in range(layers[2])])

        self.downsampler4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3,stride=(2, 1), padding=1, bias=False),
                                          nn.BatchNorm2d(256))

        self.layer4 = nn.Sequential(*[InvertBNFBlock(planes=256) for _ in range(layers[3])])

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

class DFResnet35(nn.Module):
    def __init__(self, window_length=400, hopping_length=160, mel_number=80, fft_size=256,
                 window_function=torch.hamming_window):
        super(DFResnet35, self).__init__()
        self.MelSpec = nn.Sequential(PreEmphasis(),
                                     AudioT.Spectrogram(win_length=window_length, hop_length=hopping_length,
                                                           n_mels=mel_number, n_fft=fft_size,
                                                           window_fn=window_function, sample_rate=16000))
        # self.specaug = FbankAug()

        self.instancenorm = nn.InstanceNorm1d(80)
        self.model = ResNet([2, 3, 4, 2])
        self.attention = nn.Sequential(
            nn.Conv1d(256*5, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256*5, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.fc = nn.Linear(in_features=2560, out_features=512)


    def forward(self, input_tensor: torch.Tensor):
        x = self.MelSpec(input_tensor) + 1e-6
        x = x.log()
        #  x = self.specaug(x)
        x = self.instancenorm(x).unsqueeze(1)
        x = self.model(x).reshape(x.shape[0], 1280,-1)

        w = self.attention(x)

        mu = torch.mean(x*w, dim=-1)
        sg = torch.sqrt(torch.var(x*w, dim=-1) + 1e-10)
        x = torch.cat((mu, sg), 1)

        return self.fc(x)
