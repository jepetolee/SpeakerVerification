import numpy as np
import torch, random,os
import numpy.typing as NumpyType

from tqdm import tqdm
from AAM_Softmax import makeEERScore
from torch.utils.data import DataLoader

def train(model, optimizer, loss_function, train_loader,valid_loader, epoch):
    model.train()
    train_loss,correct,total,lowest_eer = 0,0,0, 999

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
        if valid_eer < lowest_eer:
            torch.save(model.state_dict(),'./best_model.pt')

        print(f'Epoch: {iteration} | Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {100. * correct / total:.3f}')
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

    with open('./data/VoxCeleb1/trials.txt') as f:
        lines_of_test_dataset = f.readlines()

    for index, line in enumerate(lines_of_test_dataset):

        data = line.split()
        # Append random label if missing
        if len(data) == 2:
            data = [random.randint(0, 1)] + data

        ref_feat = embeddings[os.path.join('./data/VoxCeleb1/test', data[1])].cuda()
        com_feat = embeddings[os.path.join('./data/VoxCeleb1/test', data[2])].cuda()

        dist = torch.cdist(ref_feat.reshape(192, -1), com_feat.reshape(192, -1)).detach().cpu().numpy()
        score = -1 * np.mean(dist)
        all_scores.append(score)
        all_labels.append(int(data[0]))

    return makeEERScore(all_scores, all_labels)