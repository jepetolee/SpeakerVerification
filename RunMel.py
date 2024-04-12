import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader
from pytorch_multilabel_balanced_sampler.samplers import LeastSampledClassSampler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from train import train
from Model  import ResNet18_SingleChannel
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb



def RunWithArguments(batch_size=16, lr= 5.3e-4, test_name="testing everything",
                     num_epochs = 10, model_weight_decay= 2.3e-4, loss_weight_decay = 7.6e-4,
                     window_size=400, hop_size=160, window_fn=torch.hann_window, n_mel=80,
                     margin=0.2, scale=30):
    wandb.login(key="7a68c1d3f11c3c6af35fa54503409b7ff50e0312")
    wandb.init(project='SpeakerVerification Everything Resnet18 Optimization')
    num_epochs = num_epochs
    # 실행 이름 설정
    wandb.run.name = test_name + str(window_size) +"_"+str(hop_size) + "_" + window_fn.__name__+"_"+ str(n_mel)

    wandb.run.save()


    args = {
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
    }

    wandb.config.update(args, allow_val_change=True)
    model = ResNet18_SingleChannel().cuda()
    criterion = AAM_Softmax(n_class = 1211, margin=margin, scale=scale).cuda()
    criterion.train()
    optimizer =optim.AdamW(params=list(model.parameters())+list(criterion.parameters())
        ,lr=lr,weight_decay=2e-5)

    TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train",
                                   window_size=window_size,hop_size=hop_size,window_fn=window_fn,n_mel=n_mel)

    class RandomMultilabelDataset(Dataset):
        def __init__(self, *, n_examples, n_classes, mean_labels_per_example):
            class_probabilities = torch.rand([n_classes])
            class_probabilities = class_probabilities / sum(class_probabilities)
            class_probabilities *= mean_labels_per_example
            self.y = (torch.rand([n_examples, n_classes]) < class_probabilities).int()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, index):
            return {"labels": Variable(torch.tensor(self.y[index]), requires_grad=False)}

    dataset = RandomMultilabelDataset(
            n_examples=len(TrainingSet),
            n_classes=batch_size,
            mean_labels_per_example=2 )

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    sampler = LeastSampledClassSampler(labels=dataset.y,indices=indices)
    TrainDatasetLoader = DataLoader(TrainingSet, batch_size = batch_size, shuffle = False, num_workers = 10, drop_last = True,sampler=sampler)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(TrainDatasetLoader) * 30)
    ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test' ,
                              window_size=window_size,hop_size=hop_size,window_fn=window_fn,n_mel=n_mel)
    ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)
    eer = train(model, optimizer,scheduler, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/LogMel/LowestEERLogMel')
    wandb.finish()
    return eer


RunWithArguments(batch_size=128, lr= 1e-3, test_name="testing everything",
                 num_epochs = 30, model_weight_decay= 2e-5, loss_weight_decay = 2e-4,
                 window_size=400, hop_size=160, window_fn=torch.hamming_window, n_mel=80,
                 margin=0.2, scale=30)
