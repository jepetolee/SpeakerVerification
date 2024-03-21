import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader


from torch.utils.data import DataLoader
from train import train
from Model  import TestingModel

model = TestingModel().cuda()
criterion = AAM_Softmax(n_class = 1211, margin=0.2, s=30).cuda()
optimizer = optim.AdamW(model.parameters(), lr=0.02,  weight_decay=5e-4)

TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train", 300)
TrainDatasetLoader = DataLoader(TrainingSet, batch_size = 2048, shuffle = True, num_workers = 10, drop_last = True)
ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test', 300,10)
ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = True, num_workers = 10, drop_last = True)

num_epochs = 10
train(model, optimizer, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs)
