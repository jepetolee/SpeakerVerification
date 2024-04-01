#! /usr/bin/python
# -*- encoding: utf-8 -*-

import os ,random,librosa ,itertools, soundfile
import torch.nn as  nn
import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from typing import Tuple
import  torchaudio.transforms as AudioT
import torch_dct as dct
import torch
import math
# explain: DataLoader for TestData, it helps to valid function in train.py,
#          provide datas with list



class TestDataLoader(Dataset):
    def __init__(self, test_list:str, test_path:str, mel_spec=False):

        self.MelSpec = AudioT.MelSpectrogram(win_length=400, hop_length=160, n_mels=80, n_fft=412, f_min=20,
                                              f_max=7600,
                                              window_fn=torch.hamming_window, sample_rate=16000)
        self.data_list:list = list()
        self.mel_spec:bool = mel_spec

        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        set_files = list(set(files))
        set_files.sort()

        for index, line in enumerate(set_files):
            file_name = os.path.join(test_path, line)
            self.data_list.append(file_name)

    def loadWAV(self, filename):

        # Maximum audio length
        audio_length =  300 * 160 + 240

        # Read wav file and convert to torch tensor
        data, sr = librosa.load(filename, sr=16000)

        if len(data) > audio_length:
            max_offset = len(data) - audio_length
            offset = np.random.randint(max_offset)
            data = data[offset:(audio_length + offset)]

        else:
            if audio_length > len(data):
                max_offset = audio_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, audio_length - len(data) - offset), "constant")

        return data

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        data = self.loadWAV(self.data_list[index])
        data = self.MelSpec(torch.from_numpy(data))
        data = torch.log(data + 1e-8)
        if not self.mel_spec:
            data = dct.dct(data, norm='ortho')
            data = data - torch.mean(data, dim=-1, keepdim=True)
            data = data

        return data.reshape(1, 80, -1), self.data_list[index]

    def __len__(self)->int:
        return len(self.data_list)


# TrainDataBuilder (str,str,int)
# explain: DataBuilder for TrainDataset
import glob
class TrainDataBuilder(Dataset):
    def __init__(self, train_list:str, train_path:str,mel_spec:bool=False):

        self.MelSpec = AudioT.MelSpectrogram(win_length=400, hop_length=160, n_mels=80, n_fft=512, f_min=20,
                                              f_max=7600,
                                              window_fn=torch.hamming_window, sample_rate=16000)

        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join("./musan", '*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        self.rir_files = glob.glob(os.path.join("./RIRS_NOISES/simulated_rirs", '*/*/*.wav'))
        self.data_label:list = list()
        self.data_list:list = list()
        self.mel_spec:bool = mel_spec
        lines = open(train_list).read().splitlines()

        DictionaryKeys = list(set([x.split()[0] for x in lines]))
        DictionaryKeys.sort()
        DictionaryKeys = {key: index for index, key in enumerate(DictionaryKeys)}

        for index, line in enumerate(lines):
            speaker_label = DictionaryKeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)


    def loadWAV(self,filename):

        audio, sr = librosa.load(filename, sr=16000)
        length = 200 * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio], axis=0)

        augtype = random.randint(0, 6)
        if augtype < 2:  # Original
            audio = audio
        elif augtype == 2:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 3:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 4:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 5:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 6:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return audio

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        data = self.loadWAV(self.data_list[index])
        data = self.MelSpec(torch.from_numpy(data.astype(np.float32)))
        data = torch.log(data + 1e-8)
        data = AudioT.TimeMasking(time_mask_param=random.randint(0, 10))(data)
        data = AudioT.FrequencyMasking(freq_mask_param=random.randint(0, 15))(data)
        if not self.mel_spec:
            data = dct.dct(data, norm='ortho')
            data = data - torch.mean(data, dim=-1, keepdim=True)
            data = data

        return data.reshape(1, 80, -1), self.data_label[index]


    def __len__(self) -> int:
            return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = librosa.load(rir_file, sr=16000)
        rir = np.expand_dims(rir.astype(np.float32), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :200 * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = librosa.load(noise, sr=16000)
            length = 200 * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio



