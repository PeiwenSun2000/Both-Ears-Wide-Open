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


class VoxCelebMixDataset(StereoAudiowithAugDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(VoxCelebMixDataset, self).__init__(args, pr, list_sample, split)
        self.ignore_speaker = args.ignore_speaker
        self.crop_face = args.crop_face
        self.speaker_dist = self.calc_speaker_distribution()


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = os.path.join(video_path, 'audio', 'audio.wav')
        if self.crop_face:
            frame_path = os.path.join(video_path, 'faces')
        else:
            frame_path = os.path.join(video_path, 'frames')
        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['audio_sample_rate']
        frame_rate = meta_dict['frame_rate']
        frame_num = int(meta_dict['frame_num'])
        frame_list = glob.glob(f'{frame_path}/*.jpg')
        frame_list.sort()

        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        if self.split == 'train':
            if 'start_time' in info.keys():
                frame_start = int(info['start_time'])
            else:
                frame_start = np.random.choice(int(frame_num - np.ceil(ratio * self.pr.clip_length * frame_rate)), 1)[0]
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        frame_index = int(frame_start + np.floor(self.pr.clip_length * frame_rate) // 2)
        imgs = self.read_image([frame_list[frame_index]])
        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()
        
        # make audio as single channel:
        audio = torch.mean(audio, dim=0, keepdim=True)
        patch_size = int(self.pr.clip_length * audio_rate)
        lefts, rights, noaug_lefts, audios, shift_offsets = self.generate_audio(audio, audio_rate, index, patch_size)

        delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)

        itd = - shift_offsets[0] / audio_rate
        itd_neg = - shift_offsets[1] / audio_rate
        shift_offsets = np.rint(shift_offsets / self.pr.patch_stride).astype(int)
        shift_offset = 0 if self.add_mixture else shift_offsets[0]
        batch = {
            'img': imgs,
            'img_path': frame_list[frame_index],
            'audio': audios['audio'],
            'pos_audio': audios['pos_audio'],
            'neg_audio': audios['neg_audio'],
            'left_audios': lefts,
            'right_audios': rights,
            'noaug_left_audios': noaug_lefts,
            'shift_offset': torch.tensor(shift_offset),
            'audio_rate': torch.tensor(audio_rate),
            'itd': torch.tensor(itd),
            'itd_neg': torch.tensor(itd_neg),
            'delay_time': delay_time
        }
        return batch

    
    def generate_audio(self, audio, audio_rate, index, patch_size=None):
        # import pdb; pdb.set_trace()
        audio_start = 200
        max_offset = np.floor(self.pr.samp_sr * self.pr.max_delay).astype(int)
        patch_size = self.pr.patch_size if patch_size is None else patch_size

        audio_length = int(patch_size + self.pr.patch_stride * (self.pr.patch_num - 1))
        
        left_audio, right_audio, noaug_left_audio, pos_stereo_audio, neg_stereo_audio, shift_offsets = self.create_mixture_audio(audio, audio_length, max_offset, index)

        if self.aug_wave and self.split in ['train']:
            left_audio = self.augment_audio(left_audio)
            right_audio = self.augment_audio(right_audio)

        if self.add_reverb and self.split in ['train', 'val']:
            left_audio = self.addReverb(left_audio, audio_rate, index)
            right_audio = self.addReverb(right_audio, audio_rate, index)

        if not self.add_noise_with_snr == None:
            if self.split == 'train':
                snr = np.random.randint(low=0, high=self.add_noise_with_snr)
            else:
                snr = self.add_noise_with_snr
            left_audio = self.addGaussianSNR(left_audio, snr, index)
            right_audio = self.addGaussianSNR(right_audio, snr, index)

        audio = torch.cat([left_audio, right_audio], dim=0)
        lefts, rights, noaug_lefts = left_audio, right_audio, noaug_left_audio
        audios = {
            'audio': audio,
            'pos_audio': pos_stereo_audio,
            'neg_audio': neg_stereo_audio
        }
        
        return lefts, rights, noaug_lefts, audios, shift_offsets

    def select_mixture_audio(self, index_old):
        # import pdb; pdb.set_trace()
        if self.split in ['val', 'test']:
            np.random.seed(index_old)

        if self.ignore_speaker and self.split in ['train', 'val']: 
            candicates = np.arange(len(self.list_sample))
            candicates = np.delete(candicates, index_old)
            index = np.random.choice(candicates)
            info = self.list_sample[index]
        else:
            current_speaker = self.list_sample[index_old]['speaker']
            candicates = list(self.speaker_dist.keys())
            candicates.remove(current_speaker)
            new_speaker = np.random.choice(candicates)
            info = np.random.choice(self.speaker_dist[new_speaker])
            
        video_path = info['path']
        audio_path = os.path.join(video_path, 'audio', 'audio.wav')
        if self.crop_face:
            frame_path = os.path.join(video_path, 'faces')
        else:
            frame_path = os.path.join(video_path, 'frames')        
        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['audio_sample_rate']
        frame_rate = meta_dict['frame_rate']
        frame_num = int(meta_dict['frame_num'])
        frame_list = glob.glob(f'{frame_path}/*.jpg')
        frame_list.sort()
        
        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        if self.split == 'train':
            frame_start = np.random.choice(int(frame_num - np.ceil(ratio * self.pr.clip_length * frame_rate)), 1)[0]
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        frame_index = int(frame_start + np.floor(self.pr.clip_length * frame_rate) // 2)
        imgs = self.read_image([frame_list[frame_index]])
        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()
        # make audio as single channel:
        audio = torch.mean(audio, dim=0, keepdim=True)
        return {
            'img': imgs,
            'audio': audio
        }

    def create_mixture_audio(self, pos_audio, audio_length, max_offset, index):
        # import pdb; pdb.set_trace()
        neg_audio = self.select_mixture_audio(index)['audio']
        if self.split in ['val', 'test']:
            np.random.seed(index)
            
        shift_offsets = np.random.choice(max_offset * 2 + 1, 2, replace=False).astype(int) - max_offset

        pos_rms = np.maximum(1e-4, torch.sqrt(torch.mean(pos_audio**2)))
        neg_rms = np.maximum(1e-4, torch.sqrt(torch.mean(neg_audio**2)))

        if self.split == 'train':
            neg_audio = self.normalize_audio(neg_audio, desired_rms=pos_rms) 
            neg_rms = pos_rms
            alphas = torch.rand(2) + 0.5
            betas = torch.rand(2) + 0.5
            # gamma = torch.rand(1)[0] * 0.4 + 0.1
            gamma = 0
        else:
            neg_audio = self.normalize_audio(neg_audio, desired_rms=pos_rms) 
            ratios = shift_offsets / max_offset * 0.25
            alphas = [1 - ratios[0], 1 + ratios[0]]
            betas =  [1 - ratios[1], 1 + ratios[1]]
            # gamma = 0.3
            gamma = 0


        audio_start = 200
        pos_audio_left = pos_audio[:, audio_start: audio_start + audio_length]
        pos_audio_right = pos_audio[:, audio_start + shift_offsets[0]: audio_start + audio_length + shift_offsets[0]]

        neg_audio_left = neg_audio[:, audio_start: audio_start + audio_length]
        neg_audio_right = neg_audio[:, audio_start + shift_offsets[1]: audio_start + audio_length + shift_offsets[1]]

        left = self.sum2audio(alphas[0] * pos_audio_left, betas[0] * neg_audio_left)
        right = self.sum2audio(alphas[1] * pos_audio_right, betas[1] * neg_audio_right)
        noaug_left = self.sum2audio(alphas[0] * pos_audio_left, gamma * neg_audio_left)

        pos_stereo_audio = torch.cat([alphas[0] * pos_audio_left, alphas[1] * pos_audio_right], dim=0)
        neg_stereo_audio = torch.cat([betas[0] * neg_audio_left, betas[1] * neg_audio_right], dim=0)
        return (left, right, noaug_left, pos_stereo_audio, neg_stereo_audio, shift_offsets)

    def calc_speaker_distribution(self):
        speaker_dist = {}
        for item in self.list_sample:
            speaker = item['speaker']
            if not speaker in speaker_dist.keys():
                speaker_dist[speaker] = []
            speaker_dist[speaker].append(item)
        return speaker_dist