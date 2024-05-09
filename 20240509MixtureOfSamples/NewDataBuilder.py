#! /usr/bin/python
# -*- encoding: utf-8 -*-

import os ,random,librosa ,itertools, soundfile
import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from typing import Tuple
import torch
# explain: DataLoader for TestData, it helps to valid function in train.py,
#          provide datas with list

class TestDataLoader(Dataset):
    def __init__(self, test_list:str, test_path:str,n_mel=80):
        self.n_mel = n_mel

        self.data_list:list = list()

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

        # Read wav file and convert to torch tensor
        audio, sr = librosa.load(filename, sr=16000)
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:

            feats.append(audio[int(asf):int(asf) + max_audio])
        feats = np.stack(feats, axis=0).astype(np.float32)
        data = torch.FloatTensor(feats).float()

        return data

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        data = self.loadWAV(self.data_list[index])

        return data, self.data_list[index]

    def __len__(self)->int:
        return len(self.data_list)


# TrainDataBuilder (str,str,int)
# explain: DataBuilder for TrainDataset
import glob
class TrainDataBuilder(Dataset):
    def __init__(self, train_list:str, train_path:str,n_mel=80):
        self.n_mel = n_mel


        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join("../musan", '*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        self.rir_files = glob.glob(os.path.join("../RIRS_NOISES/simulated_rirs", '*/*/*.wav'))
        self.data_label:list = list()
        self.data_list:list = list()
        lines = open(train_list).read().splitlines()

        DictionaryKeys = list(set([x.split()[0] for x in lines]))
        DictionaryKeys.sort()
        DictionaryKeys = {key: index for index, key in enumerate(DictionaryKeys)}

        for index, line in enumerate(lines):
            speaker_label = DictionaryKeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)


    def loadWAV(self, filename):

        # Read wav file and convert to torch tensor

        audio, sr = librosa.load(filename, sr=16000)
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)

        past_asf = 0
        for asf in startframe:
            Addition = np.int64(random.random() * (asf-past_asf))
            past_asf = past_asf + Addition
            feats.append(audio[int(past_asf):int(past_asf) + max_audio])
        feats = np.stack(feats, axis=0).astype(np.float32)
        data = torch.FloatTensor(feats).float()

        '''        augtype = random.randint(0, 6)
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
            audio = self.add_noise(audio, 'music')'''
        return data


    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        data = self.loadWAV(self.data_list[index])

        return data, self.data_label[index]


    def __len__(self) -> int:
            return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float32), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :300 * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = 300 * 160 + 240
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



