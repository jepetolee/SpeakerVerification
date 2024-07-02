import numpy as np
import torch, random,os
import numpy.typing as NumpyType
import torch.nn as nn
import wandb
from tqdm import tqdm
from AAM_Softmax import computeEER
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from DataBuilder import TrainDataBuilder,TestDataLoader
def train(model, optimizer,scheduler, loss_function,train_loader, valid_loader, epoch,model_link):
    lowest_eer = 9999 # this is big value that EER can changes when Lowest EER marked
    #valid_eer = valid(model, valid_loader)
    scaler = GradScaler()
    for iteration in range(epoch):
        model.train()
        train_loss, correct,index =0,0,0

        # Build Train Dataset
        accumulation_steps = 8

        with tqdm(iter(train_loader)) as pbar:
            for i ,(data, targets) in enumerate(pbar):

                with autocast(device_type='cuda'):
                    inputs, targets = data.cuda(), targets.cuda()
                    outputs = model(inputs)
                    loss,acc = loss_function(outputs, targets)
                scaler.scale(loss/accumulation_steps).backward()
                if (i+1) % accumulation_steps ==0 or i == len(train_loader)-1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                correct += acc.item()
                index += len(targets)

                train_loss += loss.item()
        scheduler.step()
        valid_eer = valid(model, valid_loader)
        if valid_eer < lowest_eer:
            lowest_eer = valid_eer
            torch.save(model.state_dict(), model_link+'.pt')


        print(f'Epoch: {iteration} | Train Loss: {train_loss/len(train_loader):.3f} | Train Accuracy: {correct/index*len(targets)} ')
        print(f'Valid EER: {valid_eer:.3f}, Best EER: {lowest_eer:}')
        wandb.log({"valid_eer": valid_eer,"loss":train_loss/len(train_loader),"accuracy": correct/index*len(targets) })
    return lowest_eer


# valid
def valid(model:torch.nn.Module, valid_loader:DataLoader):
    model.eval()

    embeddings:dict = {}
    all_scores:list[NumpyType.NDArray[np.float32]] = list()
    all_labels:list[int] = list()

    with tqdm(iter(valid_loader)) as pbar:
        for input_datas, data_path in pbar:

            for data in input_datas:
                    with torch.no_grad():
                        embeddings[data_path[0]] = F.normalize(model.forward(data.cuda()), p=2, dim=1).detach().cpu()


    with open('./trials.txt') as f:
        lines_of_test_dataset = f.readlines()
    for index, line in enumerate(lines_of_test_dataset):

        data = line.split()
        # Append random label if missing
        if len(data) == 2:
            data = [random.randint(0, 1)] + data

        ref_feat = embeddings[os.path.join('./test/wav', data[1])]
        com_feat = embeddings[os.path.join('./test/wav', data[2])]

        all_scores.append(torch.mean(torch.matmul(ref_feat, com_feat.T)).detach().cpu().numpy())
        all_labels.append(int(data[0]))

    return computeEER(all_scores, all_labels)