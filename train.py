from tqdm import tqdm
from AAM_Softmax import tuneThresholdfromScore
import torch
import numpy
import random
import os

def train(model, optimizer, loss_function, train_loader,valid_loader, epoch):

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    best_eer = 9999
    for iteration in range(epoch):
        with tqdm(iter(train_loader)) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss, precision = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        valid_eer = valid(model, valid_loader)
        if valid_eer < best_eer:
            torch.save(model.state_dict(),'./best_model.pt')

        print(f'Epoch: {iteration} | Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {100. * correct / total:.3f}')
        print(f'Valid EER: {valid_eer:.3f}, Best EER: {best_eer:}')

def valid(model, valid_loader):
    model.eval()

    feats = {}
    with tqdm(iter(valid_loader)) as pbar:
        for inputs, data_path in pbar:
            inputs, data_path = inputs.cuda(), data_path
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = model.forward(inputs)
                feats[data_path[0]] = embedding_1

    with open('./data/VoxCeleb1/trials.txt') as f:
        lines = f.readlines()

    all_scores,all_labels = [],[]
    for idx, line in enumerate(lines):

        data = line.split()


        ## Append random label if missing
        if len(data) == 2:
            data = [random.randint(0, 1)] + data

        ref_feat = feats[os.path.join('./data/VoxCeleb1/test', data[1])].cuda()
        com_feat = feats[os.path.join('./data/VoxCeleb1/test', data[2])].cuda()

        dist = torch.cdist(ref_feat.reshape(192, -1), com_feat.reshape(192, -1)).detach().cpu().numpy()
        score = -1 * numpy.mean(dist)
        all_scores.append(score)
        all_labels.append(int(data[0]))
    return tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])