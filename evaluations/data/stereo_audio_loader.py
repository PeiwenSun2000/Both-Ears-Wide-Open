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

from pedalboard import (
    Pedalboard,
    Convolution,
    Compressor,
    Chorus,
    Gain,
    Reverb,
    Limiter,
    LadderFilter,
    Phaser,
)

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms

import sys
sys.path.append('..')
from utils import sound, sourcesep
from data import * 

import pdb


class StereoAudioDataset(object):
    def __init__(self, args, pr, list_sample, split='train'):
        self.pr = pr
        self.args = args
        self.setting = args.setting
        self.split = split
        self.seed = pr.seed
        self.aug_wave = args.aug_wave
        self.shift_wave = args.shift_wave
        self.regular_scaling = args.regular_scaling
        self.larger_shift = args.larger_shift

        # self.add_noise = args.add_noise
        self.no_resample = args.no_resample
        self.wav2spec = args.wav2spec
        # self.pr.clip_length = 0.96 if self.wav2spec else self.pr.clip_length
        self.fake_right = args.fake_right
        self.synthetic_rate = args.synthetic_rate if split == 'train' else 0.0
        self.normalized_rms = args.normalized_rms
        self.add_noise_with_snr = args.noiseSNR
        self.add_reverb = args.add_reverb
        self.add_mixture = args.add_mixture
        # save args parameter
        self.repeat = args.repeat if split == 'train' else 1
        self.max_sample = args.max_sample
        # self.class_info = pr.class_info

        self.image_transform = transforms.Compose(self.generate_image_transform(args, pr))
        self.video_transform = transforms.Compose(self.generate_video_transform(args, pr))

        self.list_sample = self.get_list_sample(list_sample)
        if self.max_sample > 0: 
            self.list_sample = self.list_sample[0:self.max_sample]
        
        self.data_weight = self.get_data_weight()
        if split in ['train', 'test']:
            pr.data_weight = self.data_weight

        self.list_sample = self.list_sample * self.repeat

        random.seed(self.seed)
        np.random.seed(1234)
        num_sample = len(self.list_sample)
        if self.split == 'train':
            random.shuffle(self.list_sample)
        
        # self.class_dist = self.unbalanced_dist()
        print('Audio Dataloader: # sample of {}: {}'.format(self.split, num_sample))


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = video_path+'.wav'

        
        audio_sample_rate = meta_dict['audio_sample_rate']
        frame_rate = meta_dict['frame_rate']
        frame_list = glob.glob(f'{frame_path}/*.jpg')
        frame_list.sort()
        
        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        if self.split == 'train':
            if 'start_time' in info.keys():
                frame_start = int(info['start_time'])
                random_offset = int(np.random.choice(5) - 2) 
                frame_start = frame_start + random_offset
                frame_start = max(0, frame_start)
                frame_start = min(94, frame_start)
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
        
        if_fakeright = np.random.rand() < self.synthetic_rate
        patch_size = int(self.pr.clip_length * audio_rate)
        lefts, rights, audio, shift_offset = self.generate_audio(audio, audio_rate, if_fakeright, index, patch_size)
        # if self.pr.patch_mode:

        delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)

        batch = {
            'img': imgs,
            'img_path': frame_list[frame_start],
            'audio': audio,
            'left_audios': lefts,
            'right_audios': rights,
            'audio_rate': torch.tensor(audio_rate),
            'delay_time': delay_time
            # 'fake_shift': 
        }
        return batch

    def getitem_test(self, index):
        self.__getitem__(index)

    def __len__(self): 
        return len(self.list_sample)

    def get_list_sample(self, list_sample):
        if isinstance(list_sample, str):
            samples = []
            csv_file = csv.DictReader(open(list_sample, 'r'), delimiter=',')
            for row in csv_file:
                samples.append(row)
        return samples 

    def read_audio(self, audio_path, start=0, stop=None):
        # import pdb; pdb.set_trace()
        # using soundfile
        audio, audio_rate = sf.read(audio_path, start=start, stop=stop, dtype='float64', always_2d=True)
        # repeat in case audio is too short
        if not stop == None:
            desired_audio_length = int(stop - start)
            if audio.shape[0] < desired_audio_length:
                repeat_times = np.ceil(desired_audio_length / audio.shape[0])
                audio = np.tile(audio, (int(repeat_times), 1))[:desired_audio_length, :]

        if not self.no_resample:
            if audio_rate != self.pr.samp_sr:
                audio = scipy.signal.resample(audio, int(audio.shape[0] / audio_rate * self.pr.samp_sr), axis=0)
                audio_rate = self.pr.samp_sr

        audio = np.transpose(audio, (1, 0))
        return audio, audio_rate
    
    def create_delay_matrix(self, num, audio_rate):
        delay_time = torch.arange(num).repeat(num, 1)
        offset = torch.arange(num).repeat_interleave(num).view(num, num)
        delay_time = delay_time - offset
        delay_time = delay_time * self.pr.patch_stride / audio_rate
        delay_time = delay_time.float()
        return delay_time
    
    def generate_audio(self, audio, audio_rate, if_fakeright, index, patch_size=None, add_noise=True, audio_start=None):
        # import pdb; pdb.set_trace()
        audio_start = 200 if audio_start is None else audio_start
        max_offset = np.floor(self.pr.samp_sr * self.pr.max_delay).astype(int) if self.larger_shift else 1
        patch_size = self.pr.patch_size if patch_size is None else patch_size

        audio_length = int(patch_size + self.pr.patch_stride * (self.pr.patch_num - 1))
        fake_right = if_fakeright or self.fake_right
        shift_offset = 0
        if fake_right:
            if self.split == 'train':
                shift_offset = int(np.random.choice(max_offset * 2 + 1, 1)[0] - max_offset)
                rescale = np.random.random() * 10 + 0.1
            else:
                shift_offset = int(index % (max_offset + 1) - max_offset / 2)
                rescale = ((index) % 101) / 100 + 0.5
            
            left = audio[0, audio_start: audio_start + audio_length].unsqueeze(0)
            right = audio[0, audio_start + shift_offset: audio_start + shift_offset + audio_length].unsqueeze(0) * rescale
            # audio = torch.cat([left, shift_left], dim=0)
        else:
            if self.shift_wave and self.split == 'train':
                shift_offset = int(np.random.choice(max_offset * 2 + 1, 1)[0] - max_offset) 

            left = audio[0, audio_start: audio_start + audio_length].unsqueeze(0)
            right = audio[1, audio_start + shift_offset: audio_start + audio_length + shift_offset].unsqueeze(0)

        if self.aug_wave and self.split in ['train', 'val']:
            if self.split == 'val':
                np.random.seed(index)
            left = self.augment_audio(left)
            right = self.augment_audio(right)
        
        if self.add_reverb and self.split in ['train', 'val']:
            left = self.addReverb(left, audio_rate, index)
            right = self.addReverb(right, audio_rate, index)
        
        if (not self.add_noise_with_snr == None) and add_noise:
            if self.split == 'train':
                snr = np.random.randint(low=0, high=self.add_noise_with_snr)
            elif self.split == 'val':
                np.random.seed(index)
                snr = np.random.randint(low=0, high=self.add_noise_with_snr)
            else:
                snr = self.add_noise_with_snr
            left = self.addGaussianSNR(left, snr, index)
            right = self.addGaussianSNR(right, snr, index)

        audio = torch.cat([left, right], dim=0)
        lefts, rights = left, right
        # lefts, rights = self.process_audio2patch((left, right), patch_size)
        # if self.wav2spec:
        #     lefts, rights = self.process_wav2spec((lefts, rights), audio_rate)

        return lefts, rights, audio, shift_offset

    def process_wav2spec(self, waveforms, audio_rate):
        # import pdb; pdb.set_trace()
        win_length = 256
        n_fft = self.pr.n_fft

        if self.pr.clip_length in [0.96, 2.55]:
            hop_length = 160
        else:
            sample_num = int(self.pr.clip_length * audio_rate)
            hop_length = int(sample_num // 128) 

        trans = torchaudio.transforms.Spectrogram(
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft, 
            power=None
        )
        specs = []
        for channel in waveforms:
            spec = trans(channel.squeeze(1))
            spec = spec.permute(0, 3, 2, 1)[:, :, :-1, :-1]
            specs.append(spec)

        specs = tuple(specs)
        return specs
    
    def process_wav2fft(self, waveforms, audio_rate):
        # import pdb; pdb.set_trace()
        specs = []

        for channel in waveforms:
            audio = torch.fft.rfft(channel, dim=-1)
            audio = torch.view_as_real(audio)
            audio = audio.transpose(-2, -1).squeeze(-3)
            specs.append(audio)

        specs = tuple(specs)
        return specs

    def process_audio2patch(self, waveforms, patch_size):
        # import pdb; pdb.set_trace()
        waveforms = torch.cat(waveforms, dim=0)
        waveforms = waveforms.unfold(-1, patch_size, self.pr.patch_stride)
        waveforms = waveforms.permute(1, 0, 2)
        wave_patch = [waveforms[:, i:i+1, :] for i in range(waveforms.shape[1])]
        wave_patch = tuple(wave_patch)
        return wave_patch

    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, torch.sqrt(torch.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        samples[samples > 1.] = 1.
        samples[samples < -1.] = -1.
        return samples 

    def sum2audio(self, audio_1, audio_2):
        audio = audio_1 + audio_2
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return audio

    def augment_audio(self, audio, random_factor=None):
        if random_factor == None:
            if self.regular_scaling:
                random_factor = np.random.random() + 0.5 # 0.5 - 1.5
            else:
                random_factor = 5 * np.random.random() + 0.5 # 0.5 - 5.5
        audio = audio * random_factor 
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return audio
    
    def addGaussianSNR(self, signal, snr, index):
        # import pdb; pdb.set_trace()
        signal_rms = torch.sqrt(torch.mean(signal ** 2))
        if signal_rms == 0:
            return signal
        noise_rms = torch.sqrt(signal_rms ** 2 / 10 ** (snr / 10))
        assert noise_rms > 0, f"signal rms: {signal_rms}, noise_rms: {noise_rms}, path: {self.list_sample[index]}"
        if self.split in ['test', 'val']:
            torch.manual_seed(index)
        
        noise = torch.normal(mean=0, std=noise_rms, size=signal.size())
        audio = signal + noise
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return audio

    def addReverb(self, signal, sample_rate, index, rt60=None):
        # import pdb; pdb.set_trace()
        if rt60 == None:
            if self.split == 'val':
                rt60 = (index % 100) / 100
            elif self.split == 'train':
                rt60 = torch.rand(1).item() * 0.8
        
        board = Pedalboard([Reverb(room_size=rt60), ], sample_rate=sample_rate)
        effected_signal = board(signal)
        effected_signal = torch.from_numpy(effected_signal)
        return effected_signal

    def select_mixture_audio(self, index_old):
        if self.split == 'train':
            index = np.random.randint(len(self.list_sample))
        else: 
            index = (index_old * 2) % len(self.list_sample)
        info = self.list_sample[index]
        video_path = info['path']
        audio_path = video_path+'.wav'
        frame_path = os.path.join(video_path, 'frames')
        meta_path = os.path.join(video_path, 'meta.json')

        
        audio_sample_rate = meta_dict['audio_sample_rate']
        frame_rate = meta_dict['frame_rate']
        frame_list.sort()
        
        ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
        if self.split == 'train':
            frame_start = np.random.choice(int(len(frame_list) - np.ceil(ratio * self.pr.clip_length * frame_rate)), 1)[0]
        else: 
            frame_start = int(info['start_time'])

        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)

        audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
        audio = torch.from_numpy(audio.copy()).float()
        return {
            'audio': audio
        }
    
    def calc_pgcc_phat(self, audio, audio_rate):
        # import pdb; pdb.set_trace()
        interp = 1
        # n = int(pr.clip_length * audio_rate)
        n = self.args.gcc_fft
        num = self.pr.patch_num
        if num == 1:
            step = int(audio.shape[1] - n) + 1
        else:
            step = int((audio.shape[1] - n) // (num - 1))
        audio = audio.unfold(-1, n, step)
        sig = audio[0]
        refsig = audio[1]
        n  = 2 * n
        # Generalized Cross Correlation Phase Transform
        SIG = torch.fft.rfft(sig, n=n)
        REFSIG = torch.fft.rfft(refsig, n=n)
        R = SIG * torch.conj(REFSIG)
        max_shift = int(self.pr.max_delay * audio_rate)
        pgcc_phat = []
        for i in range(11):
            beta = i * 0.1
            cc = torch.fft.irfft(R /(torch.abs(R) ** beta + 1e-20), n=(interp * n))
            cc = torch.cat([cc[:, -max_shift:], cc[:, :max_shift+1]], dim=-1).unsqueeze(1)
            pgcc_phat.append(cc)
        pgcc_phat = torch.cat(pgcc_phat, dim=1)
        pgcc_phat = pgcc_phat.unsqueeze(1)
        return pgcc_phat

    def read_image(self, frame_list):
        # import pdb; pdb.set_trace()
        imgs = []
        for img_path in frame_list:
            image = Image.open(img_path).convert('RGB')
            image = self.image_transform(image)
            imgs.append(image.unsqueeze(0))
        # (T, C, H ,W)
        imgs = torch.cat(imgs, dim=0)
        imgs = self.video_transform(imgs)
        imgs = imgs.permute(1, 0, 2, 3).squeeze()
        # (C, T, H ,W)
        return imgs
    

    def generate_image_transform(self, args, pr):
        resize_funct = transforms.Resize((pr.img_size, pr.img_size))
        vision_transform_list = [
            resize_funct,
            transforms.ToTensor(),
        ]
        return vision_transform_list

    def generate_video_transform(self, args, pr):
        if self.split == 'train':
            crop_funct = transforms.RandomCrop((pr.crop_size, pr.crop_size))
            color_funct = transforms.ColorJitter(brightness=0, contrast=0.3, saturation=0, hue=0) if args.add_color_jitter else transforms.Lambda(lambda img: img)
        else:
            crop_funct = transforms.CenterCrop((pr.crop_size, pr.crop_size))
            color_funct = transforms.Lambda(lambda img: img)

        vision_transform_list = [
            crop_funct,
            color_funct,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return vision_transform_list
    
    def get_data_weight(self):
        # default weights for binary and multi-classification
        if self.pr.num_classes == 1:
            weight = 1.0
        else:
            weight = np.array([1.0] * self.pr.num_classes) / self.pr.num_classes
        return weight
