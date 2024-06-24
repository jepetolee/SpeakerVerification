#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader
from torch.utils.data import DataLoader
from train import train
from Model.EfficientModel import ResNet34Efficient
from Model.Model import ResNet34TSTP
from Model.SpectrogramResnet import ResNet34Spectrogram
from Model.FWSEspectrogram import ResNet34FWSESpectrogram
from ContractionMappingSoftmax import CMSoftmax
from Model.DFResnet import DFResnet60,DFResnet114

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb


def RunWithArguments(testing_model,model_name,batch_size=16, lr= 5.3e-4,
                     num_epochs = 10, model_weight_decay= 2e-6,dilation=[2,2,2,2],AAM_softmax_weight_decay=2e-4,
                     window_size=400, hop_size=160, n_mel=80):
    # Setting The Project
    wandb.init(project='SpeakerVerification Researching New Model 4', settings=wandb.Settings(console='off'))
    wandb.run.name = model_name
    wandb.run.save()
    args = {
        'model_name': model_name
    }
    wandb.config.update(args, allow_val_change=True)
    '''
        'window_size':  window_size,
        'hop_size': hop_size,
        "epochs": num_epochs, 
        'window_fn': window_fn.__name__,
        "n_mel": n_mel,
        "learning_rate": lr,
        "batch_size": batch_size,
        "model_weight_decay": model_weight_decay,
        "loss_weight_decay":loss_weight_decay,
        "Scale":scale,
        "margin":margin
    '''

    # Choose Model with Pooling Type
    model = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel,dilation=dilation, fft_size=512,).cuda()
    model.load_state_dict(torch.load('./models/LogMel/LowestEERLogMelResnet34TSTP.pt'))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Add Loss, Optimizer, and Scheduler
    criterion = AAM_Softmax(n_class=1211, scale=30, margin=0.2).cuda()
    criterion.train()
    optimizer = optim.Adam(params=[{'params': model.parameters()},
                                    {'params': criterion.parameters()}],
                            lr=lr,weight_decay=model_weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-8)

    # Build Train Dataset
    TrainingSet = TrainDataBuilder('./train_list.txt', './train/wav')
    TrainDatasetLoader = DataLoader(TrainingSet, batch_size = batch_size, shuffle = True, num_workers = 10, drop_last = True)

    # Build Test Dataset
    ValidSet = TestDataLoader('./trials.txt','./test/wav')
    ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)

    #training Model for epochs and put out the lowest eer
    train(model, optimizer,scheduler, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/LogMel/LowestEERLogMel'+model_name)
    wandb.finish()
    return

if __name__ == '__main__':
    # fixing the seed Value
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wandb.login(key="7a68c1d3f11c3c6af35fa54503409b7ff50e0312")
    RunWithArguments(ResNet34TSTP, model_name='Resnet34TSTPDilation3', batch_size=16, lr=1e-4,
                     num_epochs=40, model_weight_decay=2e-5,dilation=[3,3,3,3],
                     window_size=320, hop_size=80,  n_mel=80,)

    RunWithArguments(ResNet34TSTP, model_name='Resnet34TSTPDilation2', batch_size=16, lr=1e-4,
                     num_epochs=40, model_weight_decay=2e-5,dilation=[2,2,2,2],
                     window_size=320, hop_size=80,  n_mel=80,)

