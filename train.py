import numpy as np
import torch, random,os
import numpy.typing as NumpyType
import torch.nn as nn

from tqdm import tqdm
from AAM_Softmax import computeEER
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train(model, optimizer, loss_function, train_loader,valid_loader, epoch,model_link):
    model.train()
    lowest_eer = 999
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                              T_0=20,
                                              T_mult=2)

    for iteration in range(epoch):
        train_loss, correct =0,0

        with tqdm(iter(train_loader)) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss, _ = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        valid_eer = valid(model, valid_loader)
        scheduler.step()
        if valid_eer < lowest_eer:
            lowest_eer = valid_eer
            torch.save(model.state_dict(), model_link+"_"+str(lowest_eer)+'.pt')

        print(f'Epoch: {iteration} | Train Loss: {train_loss/len(train_loader):.3f} ')
        print(f'Valid EER: {valid_eer:.3f}, Best EER: {lowest_eer:}')


# valid
def valid(model:torch.nn.Module, valid_loader:DataLoader):
    model.eval()

    embeddings:dict = {}
    all_scores:list[NumpyType.NDArray[np.float32]] = list()
    all_labels:list[int] = list()

    with tqdm(iter(valid_loader)) as pbar:
        for input_data, data_path in pbar:
            with torch.no_grad():
                embeddings[data_path[0]] = model.forward(input_data.cuda())


    cosine_similarity = nn.CosineSimilarity(dim=1,eps=1e-6)
    with open('./data/VoxCeleb1/trials.txt') as f:
        lines_of_test_dataset = f.readlines()
    for index, line in enumerate(lines_of_test_dataset):

        data = line.split()
        # Append random label if missing
        if len(data) == 2:
            data = [random.randint(0, 1)] + data

        ref_feat = embeddings[os.path.join('./data/VoxCeleb1/test', data[1])].cuda()
        com_feat = embeddings[os.path.join('./data/VoxCeleb1/test', data[2])].cuda()
        score = cosine_similarity(ref_feat,com_feat).detach().cpu().numpy()
        all_scores.append(score)
        all_labels.append(int(data[0]))

    return computeEER(all_scores, all_labels)