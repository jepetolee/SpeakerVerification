#! /usr/bin/python
# -*- encoding: utf-8 -*-

import os ,random,numpy,soundfile,torch,librosa ,itertools

import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from typing import Tuple

# explain: DataLoader for TestData, it helps to valid function in train.py,
#          provide datas with list
class TestDataLoader(Dataset):
    def __init__(self, test_list:str, test_path:str, eval_frames:int, num_eval:int):

        self.max_frames:int = eval_frames
        self.num_eval:int   = num_eval
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


    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        y, sr = librosa.load(self.data_list[index], sr=44100)
        y = self.pad_dataset(y, 44100 * 12)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
        data = np.expand_dims(mfcc, axis=-1)

        return data, self.data_list[index]

    def __len__(self)->int:
        return len(self.data_list)

    def pad_dataset(self, data, audio_length):
        # random  padding / offset
        if len(data) > audio_length:
            max_offset = len(data) - audio_length
            offset = np.random.randint(max_offset)
            data = data[offset:(audio_length + offset)]
        # pad if data is smaller
        else:
            if audio_length > len(data):
                max_offset = audio_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, audio_length - len(data) - offset), "constant")
        return data

# TrainDataBuilder (str,str,int)
# explain: DataBuilder for TrainDataset

class TrainDataBuilder(Dataset):
    def __init__(self, train_list:str, train_path:str, num_frames:int):
        self.num_frames = num_frames
        # Load and configure augmentation files
#         self.noisetypes = ['noise','speech','music']
#         self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
#         self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
#         self.noiselist = {}
#         augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
#         for file in augment_files:
#             if file.split('/')[-4] not in self.noiselist:
#                 self.noiselist[file.split('/')[-4]] = []
#             self.noiselist[file.split('/')[-4]].append(file)
#         self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

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



    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        # Read the utterance and randomly select the segment

        y, sr = librosa.load(self.data_list[index], sr=44100)
        y = self.pad_dataset(y, 44100*12)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
        data = np.expand_dims(mfcc, axis=-1)

        # Data Augmentation
#         augtype = random.randint(0,5)
#         if augtype == 0:   # Original
#             audio = audio
#         elif augtype == 1: # Reverberation
#             audio = self.add_rev(audio)
#         elif augtype == 2: # Babble
#             audio = self.add_noise(audio, 'speech')
#         elif augtype == 3: # Music
#             audio = self.add_noise(audio, 'music')
#         elif augtype == 4: # Noise
#             audio = self.add_noise(audio, 'noise')
#         elif augtype == 5: # Television noise
#             audio = self.add_noise(audio, 'speech')
#             audio = self.add_noise(audio, 'music')
        return torch.tensor(data), self.data_label[index]

    def __len__(self) -> int:
        return len(self.data_list)

    def pad_dataset(self, data, audio_length):
        # random  padding / offset
        if len(data) > audio_length:
            max_offset = len(data) - audio_length
            offset = np.random.randint(max_offset)
            data = data[offset:(audio_length + offset)]
        # pad if data is smaller
        else:
            if audio_length > len(data):
                max_offset = audio_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, audio_length - len(data) - offset), "constant")
        return data

    def add_rev(self, audio) -> np.ndarray:
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float32),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat) -> np.ndarray:
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4)
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio

