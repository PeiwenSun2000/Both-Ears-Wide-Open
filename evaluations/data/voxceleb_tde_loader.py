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


class VoxCelebMixTDEDataset(TDESimAudioDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(VoxCelebMixTDEDataset, self).__init__(args, pr, list_sample, split)
        self.crop_face = args.crop_face

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = os.path.join(video_path, 'audio.wav')

        meta_path = os.path.join(video_path, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        
        audio_sample_rate = meta_dict['Audio Sample Rate']
        frame_rate = meta_dict['frame_rate']
        speaker1_itd = meta_dict['itd1']
        speaker2_itd = meta_dict['itd2']
        angles = meta_dict['Angle']
        distances = meta_dict['Distance']
        l_mic = np.array(meta_dict['Left mic Loc']) 
        r_mic = np.array(meta_dict['Right mic Loc']) 
        mic_distance = np.sqrt(np.sum((l_mic - r_mic) ** 2))

        if self.crop_face:
            speaker1_frame_path = os.path.join(video_path, 'speaker1_faces')
            speaker2_frame_path = os.path.join(video_path, 'speaker2_faces')
        else:
            speaker1_frame_path = os.path.join(video_path, 'speaker1_frames')
            speaker2_frame_path = os.path.join(video_path, 'speaker2_frames')

        speaker1_frame_list = glob.glob(f'{speaker1_frame_path}/*.jpg')
        speaker1_frame_list.sort()

        speaker2_frame_list = glob.glob(f'{speaker2_frame_path}/*.jpg')
        speaker2_frame_list.sort()

        ratio = 1.2
        if self.split == 'train':
            if 'start_time' in info.keys():
                frame_start = int(info['start_time'])
            else:
                frame_start = np.random.choice(int(len(frame_list) - np.ceil(ratio * self.pr.clip_length * frame_rate)), 1)[0]
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        frame_index = int(frame_start + np.floor(self.pr.clip_length * frame_rate) // 2)
        speaker1_imgs = self.read_image([speaker1_frame_list[frame_index]])
        speaker2_imgs = self.read_image([speaker2_frame_list[frame_index]])

        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()
        
        if not self.add_noise_with_snr == None:
            audio = self.addGaussianSNR(audio, self.add_noise_with_snr, index)
        if_fakeright = False

        patch_size = int(self.pr.clip_length * audio_rate)
        lefts, rights, audio, shift_offset = self.generate_audio(audio, audio_rate, if_fakeright, index, patch_size, add_noise=False)

        delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)

        batch = {
            'img': speaker1_imgs,
            'img_path': speaker1_frame_list[frame_index],
            'neg_img': speaker2_imgs,
            'neg_img_path': speaker2_frame_list[frame_index],
            'audio': audio,
            'left_audios': lefts,
            'right_audios': rights,
            'audio_rate': torch.tensor(audio_rate),
            'itd': torch.tensor(speaker1_itd),
            'itd_neg': torch.tensor(speaker2_itd),
            'delay_time': delay_time,
            'shift_offset': torch.tensor(shift_offset),
            'frame_index': frame_index,
            'video_path': video_path,
            'distance': distances,
            'mic_distance': mic_distance
        }

        if self.args.setting.find('pgccphat'):
            pgcc_phat = self.calc_pgcc_phat(audio, audio_rate)
            batch['pgcc_phat'] = pgcc_phat

        if self.args.clip_length < 2.55:
            separated_sound_path = os.path.join(video_path, 'separated_sounds_short')
        else:
            separated_sound_path = os.path.join(video_path, 'separated_sounds')
        if os.path.exists(separated_sound_path) and self.args.baseline_type.find('visualvoice') != -1:
            # import pdb; pdb.set_trace()
            audio_length = audio.shape[-1]
            start = 200
            speaker1_left_path = os.path.join(separated_sound_path, 'speaker1_left.wav')
            speaker1_right_path = os.path.join(separated_sound_path, 'speaker1_right.wav')
            speaker2_left_path = os.path.join(separated_sound_path, 'speaker2_left.wav')
            speaker2_right_path = os.path.join(separated_sound_path, 'speaker2_right.wav')

            speaker1_left, audio_rate = self.read_audio(speaker1_left_path, start=audio_start, stop=audio_end)
            speaker1_left = torch.from_numpy(speaker1_left[:, start: start + audio_length].copy()).float()
            speaker1_right, audio_rate = self.read_audio(speaker1_right_path, start=audio_start, stop=audio_end)
            speaker1_right = torch.from_numpy(speaker1_right[:, start: start + audio_length].copy()).float()
            speaker2_left, audio_rate = self.read_audio(speaker2_left_path, start=audio_start, stop=audio_end)
            speaker2_left = torch.from_numpy(speaker2_left[:, start: start + audio_length].copy()).float()
            speaker2_right, audio_rate = self.read_audio(speaker2_right_path, start=audio_start, stop=audio_end)
            speaker2_right = torch.from_numpy(speaker2_right[:, start: start + audio_length].copy()).float()    

            separated_speaker1_audio = torch.cat([speaker1_left, speaker1_right], dim=0)
            separated_speaker2_audio = torch.cat([speaker2_left, speaker2_right], dim=0)
            batch['separated_speaker1_audio'] = separated_speaker1_audio
            batch['separated_speaker2_audio'] = separated_speaker2_audio

            batch['separated_speaker1_lefts'] = speaker1_left
            batch['separated_speaker1_rights'] = speaker1_right
            batch['separated_speaker2_lefts'] = speaker2_left
            batch['separated_speaker2_rights'] = speaker2_right

        return batch
