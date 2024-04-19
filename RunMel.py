import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from DataBuilder import TrainDataBuilder,TestDataLoader
from pytorch_multilabel_balanced_sampler.samplers import LeastSampledClassSampler
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from train import train
from Models.Model import ResNet34AveragePooling,ResNet34SE
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb
import random

seed = 2024
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
wandb.login(key="7a68c1d3f11c3c6af35fa54503409b7ff50e0312")
wandb.init(project='SpeakerVerification EER Of Various Models')

def RunWithArguments(testing_model,model_name,batch_size=16, lr= 5.3e-4,
                     num_epochs = 10, model_weight_decay= 2.3e-4,
                     window_size=400, hop_size=160, window_fn=torch.hann_window, n_mel=80,
                     margin=0.2, scale=30,SETYPE=None):
    if SETYPE is not None:
        model = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn, encoder_type=SETYPE).cuda()
    else:
        model = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn).cuda()

    # 실행 이름 설정
    wandb.run.name = model_name

    wandb.run.save()

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
    args = {
        'model_name': model_name
    }

    wandb.config.update(args, allow_val_change=True)

    criterion = AAM_Softmax(n_class = 1211, margin=margin, scale=scale).cuda()
    criterion.train()


    #optimizer = SAM(params=list(model.parameters())+list(criterion.parameters()),base_optimizer=optim.AdamW,lr=lr,weight_decay=2e-5)

    optimizer = optim.Adam(params=list(model.parameters())+list(criterion.parameters()),lr=lr,weight_decay=model_weight_decay)
    TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train",
                                   n_mel=n_mel)

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
    scheduler = CosineAnnealingLR(optimizer, T_max=len(TrainDatasetLoader) * (num_epochs+10))
    ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt','./data/VoxCeleb1/test' ,
                              n_mel=n_mel)
    ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)
    eer = train(model, optimizer,scheduler, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/LogMel/LowestEERLogMel')
    wandb.finish()
    return eer




RunWithArguments(ResNet34SE, 'ResNet34SE_SAP', batch_size=32, lr= 3e-4,
                 num_epochs = 20, model_weight_decay= 2e-5,
                 window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                 margin=0.2, scale=30,SETYPE='SAP')


'''
RunWithArguments(ResNet34AveragePooling, 'Resnet34AveragePooling', batch_size=32, lr= 3e-4,
                 num_epochs = 20, model_weight_decay= 2e-5,
                 window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                 margin=0.2, scale=30,SETYPE=None)
                 
RunWithArguments(ResNet34SE, 'ResNet34SE_ASP', batch_size=32, lr= 3e-4,
                 num_epochs = 20, model_weight_decay= 2e-5,
                 window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                 margin=0.2, scale=30,SETYPE='ASP')
'''