import torch
import torch.optim as optim
from AAM_Softmax import AAM_Softmax
from NewDataBuilder import TrainDataBuilder,TestDataLoader
from torch.utils.data import DataLoader
from trainSelecting import train
from Model.GatingConv import ResNet34AveragePoolingGating
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb


def RunSelectingWithArguments(testing_model,model_name,batch_size=16, lr= 5.3e-4,
                     num_epochs = 10, model_weight_decay= 2.3e-4,
                     window_size=400, hop_size=160, window_fn=torch.hann_window, n_mel=80,
                     margin=0.2, scale=30,SETYPE=None):
    wandb.init(project='SpeakerVerification Researching New Model', settings=wandb.Settings(console='off'))
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
    model.load_state_dict(torch.load('../models/LogMel/LowestEERLogMel.pt'))
    optimizer = optim.AdamW(params=[{'params':model.parameters(),'weight_decay':2e-5},
                                    {'params':criterion.parameters(),
                                     'weight_decay':3e-4}]
                            ,lr=lr)
    TrainingSet = TrainDataBuilder("./data/VoxCeleb1/train_list.txt", "./data/VoxCeleb1/train",
                                   n_mel=n_mel)
    TrainDatasetLoader = DataLoader(TrainingSet, batch_size = batch_size, shuffle = True, num_workers = 10, drop_last = True)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=35,T_mult=2,eta_min=1e-7)

    ValidSet = TestDataLoader('./data/VoxCeleb1/1111.txt','./data/VoxCeleb1/test' ,
                              n_mel=n_mel)
    ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)

    eer = train(model, optimizer,scheduler, criterion,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'./models/LogMel/LowestEERLogMel')
    wandb.finish()
    return #eer

if __name__ == '__main__':
    seed = 2024
    deterministic = True

    # andom.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.login(key="7a68c1d3f11c3c6af35fa54503409b7ff50e0312")

    RunSelectingWithArguments(ResNet34AveragePoolingGating, model_name='ResNet34AveragePoolingGating', batch_size=32 , lr=1e-4,
                     num_epochs=35, model_weight_decay=2e-5,
                     window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                     margin=0.2, scale=30, SETYPE=None)


