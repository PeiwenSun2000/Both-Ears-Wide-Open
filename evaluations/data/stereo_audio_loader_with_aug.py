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


class StereoAudiowithAugDataset(StereoAudioDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(StereoAudiowithAugDataset, self).__init__(args, pr, list_sample, split)


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = os.path.join(video_path, 'audio', 'audio.wav')
        frame_path = os.path.join(video_path, 'frames')
        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['audio_sample_rate']
        frame_rate = meta_dict['frame_rate']
        frame_list = glob.glob(f'{frame_path}/*.jpg')
        frame_list.sort()

        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        if self.split == 'train':
            if 'start_time' in meta_dict.keys():
                frame_start = int(info['start_time'])
            else:
                frame_start = np.random.choice(int(len(frame_list) - np.ceil(ratio * self.pr.clip_length * frame_rate)), 1)[0]
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        frame_index = int(frame_start + np.floor(self.pr.clip_length * frame_rate) // 2)
        imgs = self.read_image([frame_list[frame_index]])
        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()

        if_flip = np.random.rand() > 0.5 if self.split == 'train' else False

        if if_flip:
            imgs = imgs.flip(-1)
            audio = audio.flip(0)

        patch_size = int(self.pr.clip_length * audio_rate)
        
        lefts, rights, noaug_lefts, noaug_rights, audio, shift_offset = self.generate_audio(audio, audio_rate, index, patch_size)
        delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)

        batch = {
            'img': imgs,
            'img_path': frame_list[frame_start],
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

    
    def generate_audio(self, audio, audio_rate, index, patch_size=None):
        # import pdb; pdb.set_trace()
        audio_start = 200
        max_offset = np.floor(self.pr.samp_sr * self.pr.max_delay).astype(int) if self.larger_shift else 1
        patch_size = self.pr.patch_size if patch_size is None else patch_size
        audio_length = int(patch_size + self.pr.patch_stride * (self.pr.patch_num - 1))
        
        if self.shift_wave and self.split in ['train', 'val']:
            if self.split == 'train':
                shift_offset = int(np.random.choice(max_offset * 2 + 1, 1)[0] - max_offset)
            elif self.split == 'val':
                shift_offset = int((index % (max_offset + 1)) * 2 - max_offset)
        else:
            shift_offset = 0
        
        left_audio = audio[0, audio_start: audio_start + audio_length].unsqueeze(0)
        right_audio = audio[1, audio_start: audio_start + audio_length].unsqueeze(0)
        noaug_left_audio = audio[0, audio_start + shift_offset: audio_start + shift_offset + audio_length].unsqueeze(0)
        noaug_right_audio = audio[1, audio_start + shift_offset: audio_start + shift_offset + audio_length].unsqueeze(0)

        if self.add_mixture:
            left_audio, right_audio, _ = self.add_mixture_audio(left_audio, right_audio, index)

        if self.aug_wave and self.split in ['train', 'val']:
            if self.split == 'val':
                np.random.seed(index)
            left_audio = self.augment_audio(left_audio)
            right_audio = self.augment_audio(right_audio)

        if self.add_reverb and self.split in ['train', 'val']:
            left_audio = self.addReverb(left_audio, audio_rate, index)
            right_audio = self.addReverb(right_audio, audio_rate, index)

        if not self.add_noise_with_snr == None:
            if self.split == 'train':
                snr = np.random.randint(low=0, high=self.add_noise_with_snr)
            elif self.split == 'val':
                np.random.seed(index)
                snr = np.random.randint(low=0, high=self.add_noise_with_snr)
            else:
                snr = self.add_noise_with_snr
            left_audio = self.addGaussianSNR(left_audio, snr, index)
            right_audio = self.addGaussianSNR(right_audio, snr, index)

        audio = torch.cat([left_audio, right_audio], dim=0)
        lefts, rights, noaug_lefts, noaug_rights = left_audio, right_audio, noaug_left_audio, noaug_right_audio

        shift_offset = np.rint(shift_offset / self.pr.patch_stride).astype(int)
        return lefts, rights, noaug_lefts, noaug_rights, audio, shift_offset


    def add_mixture_audio(self, left, right, index):
        # import pdb; pdb.set_trace()
        neg_sample = self.select_mixture_audio(index)
        audio_start = 200
        audio_length = left.shape[-1]
        if self.split == 'train':
            scalings = torch.rand(2) * 0.9 + 0.1
        else: 
            scalings = torch.ones(2) * 0.5

        shift_offset = 0
        pos_audio = torch.cat([left, right], dim=0)
        neg_audio = neg_sample['audio'][:, audio_start: audio_start + audio_length]
        pos_rms = np.maximum(1e-4, torch.sqrt(torch.mean(pos_audio**2)))
        neg_rms = pos_rms * scalings

        neg_left = self.normalize_audio(neg_audio[0].unsqueeze(0), desired_rms=neg_rms[0]) 
        neg_right = self.normalize_audio(neg_audio[1].unsqueeze(0), desired_rms=neg_rms[1]) 

        left = left + neg_left
        right = right + neg_right
        return (left, right, neg_left)