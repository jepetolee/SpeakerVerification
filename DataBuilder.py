#! /usr/bin/python
# -*- encoding: utf-8 -*-

import os ,random,numpy,soundfile,torch,librosa ,itertools
import torch.nn as  nn
import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from typing import Tuple
import  torchaudio.transforms as AudioT
import torchvision.transforms as trans
import torchaudio
import torch

# explain: DataLoader for TestData, it helps to valid function in train.py,
#          provide datas with list
class TestDataLoader(Dataset):
    def __init__(self, test_list:str, test_path:str, eval_frames:int, num_eval:int,mel_spec=False):
        self.MelSpec = AudioT.MelSpectrogram(win_length=400, hop_length=160, n_mels=80, n_fft=512,
                                             window_fn=torch.hamming_window,sample_rate=16000)
        self.MFCC = AudioT.MFCC(sample_rate=16000, n_mfcc=80, log_mels=True, melkwargs=
        {'n_fft': 512, 'hop_length': 160, 'win_length': 400, 'n_mels': 80, "center": False})
        self.Norm = nn.InstanceNorm1d(80)
        self.max_frames:int = eval_frames
        self.num_eval:int   = num_eval
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

    def loadWAV(self,filename, max_frames, evalmode=True, num_eval=10):

        # Maximum audio length
        max_audio = max_frames * 160 + 240

        # Read wav file and convert to torch tensor
        audio, sample_rate = soundfile.read(filename)

        audiosize = audio.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = numpy.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]

        if evalmode:
            startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
        else:
            startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

        feats = []
        if evalmode and max_frames == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])

        feat = numpy.stack(feats, axis=0).astype(numpy.float32)

        return feat

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        data = self.loadWAV(self.data_list[index],  max_frames=self.max_frames,num_eval=self.num_eval)
        if not self.mel_spec:
            data = self.MFCC(torch.from_numpy(data))
            data = self.Norm(data).reshape(1, 80, 2990)
        else:
            data = self.MelSpec(torch.from_numpy(data))
            data = self.Norm(data).reshape(1, 80, 3020)

        return data, self.data_list[index]

    def __len__(self)->int:
        return len(self.data_list)


# TrainDataBuilder (str,str,int)
# explain: DataBuilder for TrainDataset

class TrainDataBuilder(Dataset):
    def __init__(self, train_list:str, train_path:str, num_frames:int,mel_spec:bool=False):
        self.num_frames = num_frames
        self.MelSpec = AudioT.MelSpectrogram(win_length=400, hop_length=160, n_mels=80,n_fft=512,window_fn=torch.hamming_window)
        self.MFCC = AudioT.MFCC(sample_rate=16000,n_mfcc=80,log_mels=True,melkwargs=
         {'n_fft': 512, 'hop_length':160,'win_length':400,'n_mels':80, "center": False})
        self.Norm = nn.InstanceNorm1d(80)
        # Load and configure augmentation files
#         self.noisetypes = ['noise','speech','music']
#         self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
#         self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
#         self.noiselist = {}
#         augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
#         for file in augment_files:es:
#             if file.split('/')[-4] not in self.noiselist:
#                 self.noiselist[file.split('/')[-4]] = []
#             self.noiselist[file.split('/')[-4]].append(file)
#         self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

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


    def loadWAV(self,filename, max_frames, evalmode=True, num_eval=10):

        # Maximum audio length
        max_audio = max_frames * 160 + 240

        # Read wav file and convert to torch tensor
        audio, sample_rate = soundfile.read(filename)

        audiosize = audio.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = numpy.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]

        if evalmode:
            startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
        else:
            startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

        feats = []
        if evalmode and max_frames == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])

        feat = numpy.stack(feats, axis=0).astype(numpy.float32)

        return feat
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        data = self.loadWAV(self.data_list[index],  max_frames=self.num_frames)

        if not self.mel_spec:
            data = self.MFCC(torch.from_numpy(data))
            data = self.Norm(data).reshape(1, 80, 2990)
        else:
            data = self.MelSpec(torch.from_numpy(data))
            data = self.Norm(data).reshape(1, 80, 3020)

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
        return data, self.data_label[index]

    def __len__(self) -> int:
        return len(self.data_list)


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

