import torchvision
import torch.nn as  nn
import torch
from Model.ResnetBaseModel import ResNetWithoutFirstLayerEncoder
import  torchaudio.transforms as AudioT
import torch.nn.functional as F
from Model.utils import PreEmphasis,FbankAug
import math

class GatingConv(nn.Module):
    def __init__(self, ):
        super(GatingConv, self).__init__()
        self.conv1 =  nn.Conv2d(1, self.inplanes, kernel_size=(7,14), stride=2, padding=3,
                               bias=False)
        self.top_k =2
        self.gate = nn.Sequential( nn.Conv2d(1, 32, kernel_size=(5,15), stride=(2,4), padding=2, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.GELU(),

                                     nn.Conv2d(32, 64, kernel_size=(5,15), stride=(2,4), padding=2, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.GELU(),

                                     nn.Conv2d(64, 128, kernel_size=(5,15), stride=(2,4), padding=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.GELU(),

                                     nn.Conv2d(128, 256, kernel_size=(5,15), stride=(2,4), padding=2, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.GELU(),

                                     nn.AdaptiveAvgPool2d((1,1)),

                                     nn.Flatten(),
                                     nn.Linear(256, 32),

                                     nn.GELU(),
                                     nn.BatchNorm1d(32),
                                     nn.Linear(32,5)
                                   )
    def forward(self, x,total):
        routing_weights =self.gate(total)

        _, selection = torch.topk( F.softmax(routing_weights, dim=1), self.top_k, dim=-1)

        selection_mask =torch.nn.functional.one_hot( torch.sort(selection,dim=1)[0], num_classes=5).permute(0, 2, 1).float()

        Gated_input = torch.einsum('BCFTN,BNS->BCFTS', x, selection_mask)

        Gated_input = Gated_input.reshape(-1, 1, 80, 2*Gated_input.shape[3])

        x = self.conv1(Gated_input)

        return x, routing_weights

class ResNet34AveragePoolingGating(nn.Module):
    def __init__(self,window_length=400,hopping_length=160, mel_number= 80, fft_size= 512, window_function=torch.hamming_window):

            super(ResNet34AveragePoolingGating, self).__init__()
            self.MelSpec = nn.Sequential(PreEmphasis(),
                                         AudioT.MelSpectrogram(win_length=window_length, hop_length=hopping_length,
                                                               n_mels=mel_number, n_fft=fft_size,
                                                               window_fn=window_function, sample_rate=16000))
            #self.specaug = FbankAug()

            self.gatingConv = GatingConv()

            self.instancenorm = nn.InstanceNorm1d(80)
            self.model = ResNetWithoutFirstLayerEncoder()
            self.fc = nn.Linear(in_features=512,out_features=507)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, input_tensors,eval=False):
            input_tensors = input_tensors.permute(1,0,2)

            MelSpectrums = []

            for  iterator in range(input_tensors.shape[0]):
                Spectrum = self.MelSpec(input_tensors[iterator]) + 1e-6
                Spectrum = Spectrum.log()
                #  x = self.specaug(x)
                Spectrum = self.instancenorm(Spectrum)
                MelSpectrums.append(Spectrum.unsqueeze(1) )

            x, Policy = self.gatingConv(torch.stack(MelSpectrums,dim=-1),torch.cat(MelSpectrums,dim=3).squeeze(2),eval=eval)
            x = self.model(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)


            return torch.cat([self.fc(x) ,Policy],dim=1)