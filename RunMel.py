import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader

from torch.utils.data import DataLoader
from train import train
from Model  import ResNet18LogMel


model = ResNet18LogMel().cuda()
model.load_state_dict(torch.load("./LogMel.pt"))

criterion = AAM_Softmax(n_class = 1211, margin=0.35, scale=30).cuda()

optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-4)
TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train", 300,mel_spec=True)
TrainDatasetLoader = DataLoader(TrainingSet, batch_size = 64, shuffle = True, num_workers = 10, drop_last = True)
ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test', 300,10,mel_spec=True)
ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)

num_epochs = 100
train(model, optimizer, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/LogMel/LowestEERLogMel')
