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


class BinauralAudioDataset(StereoAudioDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(BinauralAudioDataset, self).__init__(args, pr, list_sample, split)
        if 'itd_label' in self.list_sample[0].keys():
            self.class_dist = self.calc_class_distribution()
        # import pdb; pdb.set_trace()
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
        frame_start = int(info['start_time'])
        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        frame_index = int(frame_start + np.floor(self.pr.clip_length * frame_rate) // 2)
        imgs = self.read_image([frame_list[frame_index]])
        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()

        if not self.add_noise_with_snr == None:
            audio = self.addGaussianSNR(audio, self.add_noise_with_snr, index)

        if 'if_flip' in info.keys():
            if_flip = int(info['if_flip'])
        else:
            if_flip = np.random.rand() > 0.5 if self.split == 'train' else False

        if if_flip:
            imgs = imgs.flip(-1)
            audio = audio.flip(0)
        
        if_fakeright = np.random.rand() < self.synthetic_rate

        patch_size = int(self.pr.clip_length * audio_rate)
        lefts, rights, audio, shift_offset = self.generate_audio(audio, audio_rate, if_fakeright, index, patch_size, add_noise=False, audio_start=0)
        
        delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)

        ild_cue = self.generate_pseudo_label_from_audio(audio)
        if not 'itd_label' in info.keys(): 
            itd_label = ild_cue
        else:
            itd_label = int(info['itd_label'])

        batch = {
            'img': imgs,
            'img_path': frame_list[frame_index],
            'audio': audio,
            'left_audios': lefts,
            'right_audios': rights,
            'audio_rate': torch.tensor(audio_rate),
            'delay_time': delay_time,
            'ild_cue': torch.tensor(ild_cue),
            'itd_label': torch.tensor(itd_label),
            'if_flip': torch.tensor(int(if_flip)),
            'start_time': frame_start
        }
        if self.setting.find('pgccphat') != -1:
            pgcc_phat = self.calc_pgcc_phat(audio, audio_rate)
            batch['pgcc_phat'] = pgcc_phat
        return batch

    def generate_pseudo_label_from_audio(self, audio):
        # import pdb; pdb.set_trace()
        left_rms = torch.sqrt(torch.mean(audio[0] ** 2)).item()
        right_rms = torch.sqrt(torch.mean(audio[1] ** 2)).item()
        ratio = (left_rms - right_rms) / (left_rms + right_rms)
        if ratio > 0.05:
            pseudo_label = 0 # left
        elif ratio <= 0.05 and ratio >= 0:
            pseudo_label = 1 # middle left
        elif ratio >= -0.05 and ratio < 0:
            pseudo_label = 2 # middle right
        elif ratio < -0.05:
            pseudo_label = 3 # right
        return pseudo_label

    def calc_class_distribution(self):
        dist = {}
        for item in self.list_sample:
            label = int(item['itd_label'])
            if not label in dist.keys():
                dist[label] = []
            dist[label].append(item)
        return dist

    def generate_audio(self, audio, audio_rate, if_fakeright, index, patch_size=None, add_noise=True, audio_start=None):
        # import pdb; pdb.set_trace()
        audio_start = 200 if audio_start is None else audio_start
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

        if self.add_mixture:
            left_audio, right_audio = self.add_mixture_audio(left_audio, right_audio, index)

        if self.aug_wave and self.split in ['train', 'val']:
            if self.split == 'val':
                np.random.seed(index)
            left_audio = self.augment_audio(left_audio)
            right_audio = self.augment_audio(right_audio)

        if self.add_reverb and self.split in ['train', 'val']:
            left_audio = self.addReverb(left_audio, audio_rate, index)
            right_audio = self.addReverb(right_audio, audio_rate, index)

        if not self.add_noise_with_snr == None and add_noise:
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
        lefts, rights = left_audio, right_audio
        shift_offset = np.rint(shift_offset / self.pr.patch_stride).astype(int)
        return lefts, rights, audio, shift_offset

    def select_mixture_audio(self, index_old):
        current_label = int(self.list_sample[index_old]['itd_label'])
        candicates = []
        for label in self.class_dist.keys():
            if label == current_label:
                continue
            candicates += self.class_dist[label]
        
        np.random.seed(index_old)
        info = np.random.choice(candicates)
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
            frame_start = np.random.choice(int(len(frame_list) - np.ceil(ratio * self.pr.clip_length * frame_rate)), 1)[0]
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        frame_index = int(frame_start + np.floor(self.pr.clip_length * frame_rate) // 2)
        imgs = self.read_image([frame_list[frame_index]])
        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()
        return {
            'img': imgs,
            'audio': audio
        }

    def add_mixture_audio(self, left, right, index):
        # import pdb; pdb.set_trace()
        neg_sample = self.select_mixture_audio(index)
        audio_start = 0
        audio_length = left.shape[-1]
        neg_audio = neg_sample['audio'][:, audio_start: audio_start + audio_length]

        left = left + neg_audio[0].unsqueeze(0)
        right = right + neg_audio[1].unsqueeze(0)
        return (left, right)