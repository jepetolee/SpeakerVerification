import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader

from torch.utils.data import DataLoader
from train import train
from Model  import ResNet18MFCC80


num_epochs = 44
model = ResNet18MFCC80().cuda()
criterion = AAM_Softmax(n_class = 1211, margin=0.35, scale=30).cuda()

optimizer = optim.AdamW([
    {'params': model.parameters()},
    {'params': criterion.parameters(), 'weight_decay': 2e-4}
], lr=1e-3, weight_decay=2e-5)

TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train")
TrainDatasetLoader = DataLoader(TrainingSet, batch_size = 128, shuffle = True, num_workers = 10, drop_last = True)
ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test')
ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)
train(model, optimizer, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/MFCC/LowestEERMFCC80Channel')
