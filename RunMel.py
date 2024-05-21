#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader
from torch.utils.data import DataLoader
from train import train
from Model.Model import ResNet34AveragePooling,ResNet34SE ,ResNet34SEPointwise,ResNet34DoubleAttention
from Model.ResnetFWSE import ResNet34FWSE
from Model.FWSEspectrogram import ResNet34FWSESpectrogram
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb


def RunWithArguments(testing_model,model_name,batch_size=16, lr= 5.3e-4,
                     num_epochs = 10, model_weight_decay= 2e-6,AAM_softmax_weight_decay=2e-4,
                     window_size=400, hop_size=160, window_fn=torch.hann_window, n_mel=80,
                     margin=0.2, scale=30,SETYPE=None):
    # Setting The Project
    wandb.init(project='SpeakerVerification Researching New Model', settings=wandb.Settings(console='off'))
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
    if SETYPE is not None:
        model = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn, encoder_type=SETYPE).cuda()
    else:
        model = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn).cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Add Loss, Optimizer, and Scheduler
    criterion = AAM_Softmax(n_class=1211, margin=margin, scale=scale).cuda()
    criterion.train()
    optimizer = optim.AdamW(params=[{'params': model.parameters(), 'weight_decay': model_weight_decay},
                                    {'params': criterion.parameters(), 'weight_decay': AAM_softmax_weight_decay}],
                            lr=lr)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-7)

    # Build Train Dataset
    TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train",n_mel=n_mel)
    TrainDatasetLoader = DataLoader(TrainingSet, batch_size = batch_size, shuffle = True, num_workers = 10, drop_last = True)

    # Build Test Dataset
    ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test' ,n_mel=n_mel)
    ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)

    #training Model for epochs and put out the lowest eer
    eer = train(model, optimizer,scheduler, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/LogMel/LowestEERLogMel'+model_name)
    wandb.finish()
    return eer

if __name__ == '__main__':
    # fixing the seed Value
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wandb.login(key="7a68c1d3f11c3c6af35fa54503409b7ff50e0312")

    # Running Functions, this can be expressed with Kind Of Tests
    RunWithArguments(ResNet34FWSESpectrogram, model_name='ResNet34FWSESpectrogram-pre-Emphasis', batch_size=16, lr=1e-4,
                     num_epochs=30, model_weight_decay=2e-5,
                     window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                     margin=0.2, scale=30, SETYPE=None)


    RunWithArguments(ResNet34FWSE, model_name='ResNet34FWSE_no-pre-Emphasis', batch_size=32, lr=3e-4,
                     num_epochs=30, model_weight_decay=2e-5,
                     window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                     margin=0.2, scale=30, SETYPE=None)

    RunWithArguments(ResNet34AveragePooling, model_name='ResNet34AveragePooling_no-pre-Emphasis', batch_size=32, lr=3e-4,
                     num_epochs=30, model_weight_decay=2e-5,
                     window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                     margin=0.2, scale=30, SETYPE=None)


