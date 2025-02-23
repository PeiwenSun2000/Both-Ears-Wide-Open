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


class TDESimAudioDataset(StereoAudioDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(TDESimAudioDataset, self).__init__(args, pr, list_sample, split)
        self.max_weight = args.max_weight

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = os.path.join(video_path, 'audio.wav')
        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['Audio Sample Rate']
        audio_length = meta_dict['Audio Length']
        angle = meta_dict['Angle']
        itd = meta_dict['itd']
        frame_rate = 10
        frame_num = int(audio_length * frame_rate)

        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        # clip_length = 1
        clip_length = ratio * self.pr.clip_length
        if self.args.same_noise:
            clip_length = 1

        if self.split == 'train':
            length = int(frame_num - np.ceil(clip_length * frame_rate))
            frame_start = np.random.choice(length, 1)[0] if length > 0 else int(0)
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + clip_length * audio_sample_rate)

        # imgs = self.read_image([frame_path])
        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()

        if self.add_mixture:
            audio = self.add_mixture_audio(audio, index)

        if not self.add_noise_with_snr == None:
            if self.split == 'train':
                snr = np.random.randint(low=0, high=self.add_noise_with_snr)
            elif self.split == 'val':
                np.random.seed(index)
                snr = np.random.randint(low=0, high=self.add_noise_with_snr)
            else:
                snr = self.add_noise_with_snr
            audio = self.addGaussianSNR(audio, self.add_noise_with_snr, index)

        if_flip = np.random.rand() > 1 if self.split == 'train' else False

        if if_flip:
            audio = audio.flip(0)
        
        if_fakeright = np.random.rand() < self.synthetic_rate

        patch_size = int(self.pr.clip_length * audio_rate)
        lefts, rights, audio, shift_offset = self.generate_audio(audio, audio_rate, if_fakeright, index, patch_size, add_noise=False)
        
        delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)

        if self.fake_right:
            itd = - shift_offset / audio_rate

        itd_in_sample = itd * audio_rate

        batch = {
            # 'img': imgs,
            # 'img_path': frame_path,
            'audio': audio,
            'left_audios': lefts,
            'right_audios': rights,
            'itd': torch.tensor(itd),
            'audio_rate': torch.tensor(audio_rate),
            'delay_time': delay_time,
            'angle': torch.tensor(angle),
            'itd_in_sample': torch.tensor(itd_in_sample)
        }
        if self.setting.find('pgccphat') != -1:
            pgcc_phat = self.calc_pgcc_phat(audio, audio_rate)
            batch['pgcc_phat'] = pgcc_phat
        return batch

    def select_mixture_audio(self, cur_index):
        # import pdb; pdb.set_trace()
        if self.split in ['train', 'val']:
            info = np.random.choice(self.list_sample)
            video_path = info['path']
        else:
            info = self.list_sample[cur_index]
            video_path = info['path']
            cur_sample_idx = int(video_path[-4:])
            np.random.seed(cur_index)
            cur_set = int(cur_sample_idx // 2100)
            new_sample_idx = np.random.randint(2100 * cur_set, 2100 * (cur_set+1))
            video_path = video_path[:-4] + str(new_sample_idx).zfill(4)

        audio_path = os.path.join(video_path, 'audio.wav')
        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['Audio Sample Rate']
        audio_length = meta_dict['Audio Length']
        angle = meta_dict['Angle']
        itd = meta_dict['itd']
        frame_rate = 10
        frame_num = int(audio_length * frame_rate)

        # ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        clip_length = ratio * self.pr.clip_length
        if self.args.same_noise:
            clip_length = 1
        if self.split == 'train':
            length = int(frame_num - np.ceil(clip_length * frame_rate))
            frame_start = np.random.choice(length, 1)[0] if length > 0 else int(0)
        else: 
            frame_start = 1
        
        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + clip_length * audio_sample_rate)

        # imgs = self.read_image([frame_path])
        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()

        return {
            'audio': audio
        }

    def add_mixture_audio(self, audio, index):
        # import pdb; pdb.set_trace()
        neg_audio = self.select_mixture_audio(index)['audio']

        if self.split in ['train', 'val']:
            scalings = np.random.rand() * 0.9
        else:
            np.random.seed(index)
            scalings = self.max_weight
        
        pos_rms = np.maximum(1e-4, torch.sqrt(torch.mean(audio**2)))
        neg_rms = pos_rms * scalings
        neg_audio = self.normalize_audio(neg_audio, desired_rms=neg_rms) 
        shortest = min(audio.shape[1], neg_audio.shape[1])
        audio = audio[:, :shortest] + neg_audio[:, :shortest]
        return audio
    