import os

import numpy as np

import sys
sys.path.append('..')
from utils import utils

Params = utils.Params


def base(name):
    pr = Params(
        frame_rate = 10,
        samp_sr = 16000,
        clip_length = 0.96,
        log_spec = True,
        f_min=0,
        f_max=None,
        log_offset=1e-10,
        n_mel = 128,
        hop_length = 0.01,
        win_length= 0.016,
        n_fft=256,
        spec_min=-100.,
        spec_max = 100.,
        num_samples = 0,
        # mono = True,
        seed=1234,
        img_size=256,
        crop_size=224,
        flip_prob = 0.5,
        gamma=0.3,
        num_classes=1,
        feat_dim = 128,
        objective=None,
        net=None,
        dataloader=None,
        loss=None,
        format='mel',
        lr_milestones = [20, 40, 60, 80, 100],
        # list_train = 'data/Youtube-ASMR/data-split/keytime/train.csv',
        list_train = 'data/Youtube-ASMR/data-split/train.csv',
        list_val = 'data/Youtube-ASMR/data-split/keytime/val.csv',
        list_test = 'data/Youtube-ASMR/data-split/keytime/test.csv',
        list_vis = 'data/Youtube-ASMR/data-split/vis_test.csv',
        fake_right=False,
        patch_mode=False,
        patch_size=512,
        patch_stride=4,
        patch_num=12,
        max_delay=1.0e-3,
        weight_key=None,
        head_width=0.18,
        radius=1.5,
        sound_speed=340,
        tau=0.05
    )

    return pr




# --------------------- Patch ------------------------ # 

def stereocrw_asmr(**kwargs):
    pr = base('stereo_crw', **kwargs)
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.num_classes = 1
    pr.feat_dim = 128
    pr.dataloader = 'StereoAudiowithAugDataset'
    pr.net = 'WaveNet'
    pr.loss = 'StereoCRWLoss'
    pr.patch_size = 7680
    pr.patch_stride = 4
    pr.patch_num = 49
    pr.patch_mode = True
    return pr


def aug_stereocrw_asmr(**kwargs):
    pr = base('stereo_crw', **kwargs)
    pr.clip_length = 0.48
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.num_classes = 1
    pr.feat_dim = 128
    pr.dataloader = 'StereoAudiowithAugDataset'
    pr.net = 'WaveAugNet'
    pr.loss = 'StereoCRWAugLoss'
    pr.patch_size = 7680
    pr.patch_stride = 4
    pr.patch_num = 49
    pr.patch_mode = True
    return pr

def stereonce_asmr(**kwargs):
    pr = stereocrw_asmr(**kwargs)
    pr.loss = 'StereoNCELoss'
    return pr


def stereocrw_fairplay(**kwargs):
    pr = stereocrw_asmr(**kwargs)
    # pr.samp_sr = 48000
    pr.list_train = 'data/FAIR-Play/data-split/train.csv'
    pr.list_val = 'data/FAIR-Play/data-split/val.csv'
    pr.list_test = 'data/FAIR-Play/data-split/test.csv'
    pr.list_vis = 'data/FAIR-Play/data-split/vis_test.csv'
    return pr

def stereonce_fairplay(**kwargs):
    pr = stereocrw_fairplay(**kwargs)
    pr.loss = 'StereoNCELoss'
    return pr

def aug_stereocrw_fairplay(**kwargs):
    pr = aug_stereocrw_asmr(**kwargs)
    # pr.samp_sr = 48000
    pr.list_train = 'data/FAIR-Play/data-split/train.csv'
    pr.list_val = 'data/FAIR-Play/data-split/val.csv'
    pr.list_test = 'data/FAIR-Play/data-split/test.csv'
    return pr



def stereocrw_freemusic(**kwargs):
    pr = stereocrw_asmr(**kwargs)
    pr.dataloader = 'FreeMusicwithAugDataset'
    pr.list_train = 'data/Free-Music-Archive/data-split/train.csv'
    pr.list_val = 'data/Free-Music-Archive/data-split/val.csv'
    pr.list_test = 'data/Free-Music-Archive/data-split/test.csv'
    return pr


def stereonce_freemusic(**kwargs):
    pr = stereocrw_freemusic(**kwargs)
    pr.loss = 'StereoNCELoss'
    return pr


def aug_stereocrw_freemusic(**kwargs):
    pr = aug_stereocrw_asmr(**kwargs)
    pr.dataloader = 'FreeMusicwithAugDataset'
    pr.list_train = 'data/Free-Music-Archive/data-split/train.csv'
    pr.list_val = 'data/Free-Music-Archive/data-split/val.csv'
    pr.list_test = 'data/Free-Music-Archive/data-split/test.csv'
    return pr



def stereocrw_binaural(**kwargs):
    pr = stereocrw_asmr(**kwargs)
    pr.max_delay = 1.5 * 1e-3
    pr.dataloader = 'BinauralAudioDataset'
    pr.list_test = 'data/Youtube-Binaural/data-split/in-the-wild/test_with_label.csv'
    return pr



def stereocrw_tdesim(**kwargs):
    pr = stereocrw_asmr(**kwargs)
    # pr.samp_sr = 48000
    pr.max_delay = 1.5 * 1e-3
    pr.dataloader = 'TDESimAudioDataset'
    pr.list_test = 'data/TDE-Simulation/data-split/test.csv'
    return pr


# --------------------- Audio-Visual model ------------------------ # 

def voxceleb_audio_visual_augstereocrw(**kwargs):
    pr = base('stereo_crw', **kwargs)
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.num_classes = 1
    pr.feat_dim = 128
    pr.frame_rate = 5
    pr.img_size = 224
    pr.crop_size = 224
    pr.dataloader = 'VoxCelebMixDataset'
    pr.net = 'AudioVisualAugCRWNet'
    pr.loss = 'VoxCelebAVITDLoss'
    pr.patch_size = 7680
    pr.patch_stride = 4
    pr.patch_num = 49
    pr.patch_mode = True
    pr.max_delay = 1.5e-3
    pr.list_train = 'data/VoxCeleb2/data-split/Newspeaker_ITD/train.csv'
    pr.list_val = 'data/VoxCeleb2/data-split/Newspeaker_ITD/val.csv'
    pr.list_test = 'data/VoxCeleb2/data-split/voxceleb-tde/Easy/test.csv'
    return pr

def voxcelebtde_audio_visual_stereocrw(**kwargs):
    pr = voxceleb_audio_visual_augstereocrw(**kwargs)
    pr.dataloader = 'VoxCelebMixTDEDataset'
    pr.net = 'AudioVisualCRWNet'
    pr.loss = 'VoxCelebAVITDLoss'
    return pr

def voxceleb_stereocrw(**kwargs):
    pr = voxceleb_audio_visual_augstereocrw(**kwargs)
    pr.net = 'WaveNet'
    return pr

def aug_stereocrw_voxceleb(**kwargs):
    pr = voxceleb_audio_visual_augstereocrw(**kwargs)
    pr.net = 'WaveAugNet'
    pr.loss = 'StereoCRWAugLoss'
    return pr

def voxcelebtde_stereocrw(**kwargs):
    pr = voxceleb_audio_visual_augstereocrw(**kwargs)
    pr.net = 'WaveNet'
    pr.dataloader = 'VoxCelebMixTDEDataset'
    return pr
