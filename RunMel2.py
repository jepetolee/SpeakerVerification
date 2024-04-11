import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader
from pytorch_multilabel_balanced_sampler.samplers import LeastSampledClassSampler

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from train import train
from Resnet34  import SE_ResNet_Encoder

from SAM import  SAM
import numpy as np
num_epochs = 48
import wandb
wandb.login(key="7a68c1d3f11c3c6af35fa54503409b7ff50e0312")
wandb.init(project='SpeakerVerification Resnet18')

# 실행 이름 설정
wandb.run.name = 'Wandb-SpeakerVerification Resnet34'
wandb.run.save()

args = {
    "learning_rate": 1e-3,
    "epochs": num_epochs,
    "batch_size": 128
}
wandb.config.update(args)

model = SE_ResNet_Encoder().cuda()
criterion = AAM_Softmax(n_class = 1211, margin=0.2, scale=30).cuda()


optimizer =optim.Adam([
    {'params': model.parameters(),'weight_decay': 2e-5},
    {'params': criterion.parameters(),'weight_decay': 2e-4}]
    ,lr=1e-3)



TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train",mel_spec=True)

class RandomMultilabelDataset(Dataset):
    def __init__(self, *, n_examples, n_classes, mean_labels_per_example):
        class_probabilities = torch.rand([n_classes])
        class_probabilities = class_probabilities / sum(class_probabilities)
        class_probabilities *= mean_labels_per_example
        self.y = (torch.rand([n_examples, n_classes]) < class_probabilities).int()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"labels": Variable(torch.tensor(self.y[index]), requires_grad=False)}

dataset = RandomMultilabelDataset(
        n_examples=len(TrainingSet),
        n_classes=128,
        mean_labels_per_example=2,
    )
indices = list(range(len(dataset)))
np.random.shuffle(indices)
sampler = LeastSampledClassSampler(labels=dataset.y,indices=indices)

TrainDatasetLoader = DataLoader(TrainingSet, batch_size = 128, shuffle = False, num_workers = 10, drop_last = True,sampler=sampler)

ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test' ,mel_spec=True)
ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)
train(model, optimizer, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/LogMel/LowestEERLogMel')
