import torch
from DataBuilder import TestDataLoader
from torch.utils.data import DataLoader
from train import valid
from Model.Model import ResNet34TSTP
from Model.GroupLearningVersion1 import GroupLearning
import wandb
import os ,random,itertools, soundfile
import numpy as np
def RunWithArguments(testing_model,model_name,batch_size=16, lr= 5.3e-4,
                     num_epochs = 10, model_weight_decay= 2e-6,AAM_softmax_weight_decay=2e-4,
                     window_size=400, hop_size=160, window_fn=torch.hann_window, n_mel=80,
                     margin=0.2, scale=30,SETYPE=None):
    # Setting The Project
    wandb.init(project='Testing Group Learning', settings=wandb.Settings(console='off'))
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
        model1 = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn, encoder_type=SETYPE).cuda()
        model2 = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn, encoder_type=SETYPE).cuda()
        model3 = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn, encoder_type=SETYPE).cuda()
    else:
        model1 = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn).cuda()
        model2 = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn).cuda()
        model3 = testing_model(window_length=window_size, hopping_length=hop_size, mel_number=n_mel, fft_size=512,
                              window_function=window_fn).cuda()

    model1.load_state_dict(torch.load("./models/LogMel/LowestEERLogMelResnet34AveragePooling.pt"))
    model2.load_state_dict(torch.load("./models/LogMel/LowestEERLogMelResnet34AveragePooling.pt"))
    model3.load_state_dict(torch.load("./models/LogMel/LowestEERLogMelResnet34AveragePooling.pt"))
    model = GroupLearning(model1,model2,model3).cuda()
    # Add Loss, Optimizer, and Scheduler

   # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)


    with open("./data/VoxCeleb1/train_list.txt") as f:
      lines = f.readlines()

    ## Get a list of unique file names
    files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
    set_files = list(set(files))
    set_files.sort()

    DictionaryKeys = list(set([x.split()[0] for x in lines]))
    DictionaryKeys.sort()
    DictionaryKeys = {key: index for index, key in enumerate(DictionaryKeys)}
    group_dict = {v: list() for k, v in DictionaryKeys.items()}
    for index, line in enumerate(lines):
        speaker_label = DictionaryKeys[line.split()[0]]
       # print(speaker_label)
        file_name = os.path.join("./data/VoxCeleb1/train", line.split()[1])
        group_dict[speaker_label].append(file_name)

    ValidSet = TestDataLoader('./data/VoxCeleb1/trials.txt', './data/VoxCeleb1/test', n_mel=n_mel)
    ValidDatasetLoader = DataLoader(ValidSet, batch_size=1, shuffle=False, num_workers=10, drop_last=True)

    model.train()
    train_loss, index = 0,  0
    training_step = 0
    lowest_eer = 9999
    while True:
        random_numbers = random.sample(range(1211), 16)

        stacking_number = []
        for index in random_numbers:
            random_value = random.sample(group_dict[index], 2)

            couple_one = []
            for indexOfRandomValue in random_value:
                audio, sr = soundfile.read(indexOfRandomValue)
                length = 300 * 160 + 240  # Our Sample Time(Sec) for BaseLine is 3 seconds(16000(sr) *3)
                if audio.shape[0] <= length:
                    shortage = length - audio.shape[0]
                    audio = np.pad(audio, (0, shortage), 'wrap')
                start_frame = np.int64(random.random() * (audio.shape[0] - length))
                audio = audio[start_frame:start_frame + length]
                audio = np.stack([audio], axis=0)
                audio = torch.tensor(audio[0]).float()
                couple_one.append(audio)

            stacking_number.append(torch.stack(couple_one, dim=0))
        stacked_data = torch.stack(stacking_number, dim=0).permute(1, 0, 2).cuda()

        loss = model.TrainLoss(stacked_data)

        index += 20*2
        train_loss += loss.item()

        training_step+=1
        if training_step%2000==1:

            valid_eer = valid(model, ValidDatasetLoader)
            if valid_eer < lowest_eer:
                lowest_eer = valid_eer
                torch.save(model.state_dict(), './models/LogMel/LowestEERLogMel' + model_name + '.pt')
            print(f'Epoch: {training_step} | Train Loss: {train_loss / index:.3f} ')
            print(f'Valid EER: {valid_eer:.3f}, Best EER: {lowest_eer:}')
            wandb.log({"valid_eer": valid_eer,"loss":train_loss/index})
            train_loss, index = 0, 0
    wandb.finish()
    return


if __name__ == '__main__':
    # fixing the seed Value
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wandb.login(key="7a68c1d3f11c3c6af35fa54503409b7ff50e0312")

    # Running Functions, this can be expressed with Kind Of Tests
    RunWithArguments(ResNet34TSTP, model_name='GroupLearning_1', batch_size=32, lr=1e-4,
                     num_epochs=35, model_weight_decay=2e-5,
                     window_size=320, hop_size=80, window_fn=torch.hamming_window, n_mel=80,
                     margin=0.2, scale=30, SETYPE=None)