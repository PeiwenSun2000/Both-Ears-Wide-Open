import csv
import glob
import h5py
import io
import json
import librosa
import numpy as np
import os
import pickle
from PIL import Image
from PIL import ImageFilter
import random
import scipy
import soundfile as sf
import time
from tqdm import tqdm
import glob
import cv2

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms

import sys
sys.path.append('..')
from utils import sound, sourcesep
from data import * 

import pdb


class FreeMusicwithAugDataset(StereoAudiowithAugDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(FreeMusicwithAugDataset, self).__init__(args, pr, list_sample, split)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = os.path.join(video_path, 'audio.wav')
        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['audio_sample_rate']
        audio_length = meta_dict['audio_length']
        frame_rate = 10
        frame_num = int(audio_length * frame_rate) - 1

        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        if self.split == 'train':
            length = int(frame_num - np.ceil(ratio * self.pr.clip_length * frame_rate))
            frame_start = np.random.choice(length, 1)[0] if length > 0 else int(0)
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()

        if_flip = np.random.rand() > 0.5 if self.split == 'train' else False

        if if_flip:
            audio = audio.flip(0)
        # import pdb; pdb.set_trace()
        patch_size = int(self.pr.clip_length * audio_rate)
        lefts, rights, noaug_lefts, noaug_rights, audio, shift_offset = self.generate_audio(audio, audio_rate, index, patch_size)
        delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)

        batch = {
            'audio': audio,
            'left_audios': lefts,
            'right_audios': rights,
            'noaug_left_audios': noaug_lefts,
            'noaug_right_audios': noaug_rights,
            'shift_offset': torch.tensor(shift_offset),
            'audio_rate': torch.tensor(audio_rate),
            'delay_time': delay_time
        }
        return batch
    
    def select_mixture_audio(self, index_old):
        if self.split == 'train':
            index = np.random.randint(len(self.list_sample))
        else: 
            index = (index_old * 2) % len(self.list_sample)
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = os.path.join(video_path, 'audio.wav')
        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['audio_sample_rate']
        audio_length = meta_dict['audio_length']
        frame_rate = 10
        frame_num = int(audio_length * frame_rate) - 1

        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2

        if self.split == 'train':
            length = int(frame_num - np.ceil(ratio * self.pr.clip_length * frame_rate))
            frame_start = np.random.choice(length, 1)[0] if length > 0 else int(0)
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()
        return {
            'audio': audio
        }