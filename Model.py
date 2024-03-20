import torchvision
import torch.nn as  nn
import torch

class TestingModel(nn.Module):
    def __init__(self):
            super(TestingModel, self).__init__()
            self.Reader = nn.Conv2d(1,3,kernel_size =1,stride=1,padding=0)
            self.aa = torchvision.models.resnet18(pretrained=False)
            self.Classifier = nn.Sequential(nn.Linear(1000, 192))

    def forward(self, input_tensor: torch.Tensor):
            x =self.Reader(input_tensor.reshape(-1, 1, 40, 13))
            x = self.aa(x)
            return self.Classifier(x)