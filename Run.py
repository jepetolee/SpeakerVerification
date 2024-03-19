import torch.optim as optim
from AAM_Softmax import AAMsoftmax
from DataBuilder import DataBuilder
import os
from torch.utils.data import DataLoader
from train import train
from Model  import TestingModel

model = TestingModel().cuda()
criterion = AAMsoftmax(n_class = 1211,m=0.2,s=30).cuda()
optimizer = optim.AdamW(model.parameters(), lr=0.02,  weight_decay=5e-4)

train_list = "./data/VoxCeleb1/train_list.txt"
train_path = "./data/VoxCeleb1/train"


TrainingSet = DataBuilder(train_list, train_path, 300)
TrainDatasetLoader = DataLoader(TrainingSet, batch_size = 2048, shuffle = True, num_workers = 10, drop_last = True)

ValidSet = DataBuilder('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test', 300)
ValidDatasetLoader = DataLoader(ValidSet, batch_size = 2048, shuffle = True, num_workers = 10, drop_last = True)

num_epochs = 10
train(model, optimizer, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs)

'''
testset = train_loader("./data/VoxCeleb1/train_list.txt","./data/VoxCeleb1/test",None,None,300)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
'''