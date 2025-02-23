import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np
import time
import csv
from tqdm import tqdm
from collections import OrderedDict
import soundfile as sf
import imageio
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
import shutil

from config import init_args, params
# from data import *
import data
import models
from models import *
from utils import utils, torch_utils


def update_param(args, pr):
    for attr in vars(pr).keys():
        if attr in vars(args).keys():
            attr_args = getattr(args, attr)
            if attr_args is not None:
                setattr(pr, attr, attr_args)
    if args.crop_face:
        pr.img_size = 160
        pr.crop_size = 160

def write_csv(data_list, filepath):
    # import pdb; pdb.set_trace()
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for info in data_list:
            writer.writerow(info)
    # print('{} items saved to {}.'.format(len(data_list), filepath))


def predict(args, pr, net, batch, device, evaluate=False, loss=False):
    # import pdb; pdb.set_trace()
    inputs = {}
    inputs['left_audios'] = batch['left_audios'].to(device)
    inputs['right_audios'] = batch['right_audios'].to(device)
    inputs['delay_time'] = batch['delay_time'].to(device)
    
    if args.setting.find('aug') != -1:
        inputs['noaug_left_audios'] = batch['noaug_left_audios'].to(device)
        inputs['shift_offset'] = batch['shift_offset'].to(device)

    if args.setting.find('audio_visual') != -1:
        inputs['img'] = batch['img'].to(device)
        inputs['shift_offset'] = batch['shift_offset'].to(device)
    
    out = net(inputs, evaluate=evaluate, loss=loss)
    return out


def mask_by_max_delay_window(args, pr, aff):
    max_delay_offset = np.floor(pr.samp_sr * pr.max_delay / pr.patch_stride).astype(int)
    N, H, W = aff.shape
    repeats = torch.eye(H, W).view(-1).long() * (max_delay_offset * 2) + 1
    mask = torch.eye(H, W).view(-1).repeat_interleave(repeats, dim=-1).view(H, -1)
    mask = mask.repeat(N, 1, 1).to(aff.device)
    mask = mask[:, :, max_delay_offset : -max_delay_offset]
    masked_aff = aff * mask
    masked_aff[masked_aff == 0] = -1e20
    masked_aff = F.softmax(masked_aff, dim=-1)
    return masked_aff


def inference_crw_itd_simple(args, pr, aff_L2R, delay_time):
    # import pdb; pdb.set_trace()
    max_delay_offset = np.floor(pr.samp_sr * pr.max_delay / pr.patch_stride).astype(int)
    masked_aff = mask_by_max_delay_window(args, pr, aff_L2R)
    if args.select == 'soft_weight':
        crw_itd = torch.sum(masked_aff * delay_time, dim=-1)
    elif args.select == 'argmax':
        B, N, M = masked_aff.size()
        aff = masked_aff.contiguous().view(B * M, -1)
        inds_y = torch.argmax(aff, dim=-1)
        inds_x = torch.arange(aff.shape[0]).to(aff.device)
        crw_itd = delay_time.view(B * M, -1)[inds_x, inds_y]
        crw_itd = crw_itd.view(B, N)

    crw_itd = crw_itd[:, max_delay_offset: -max_delay_offset]

    if args.bidirectional:
        masked_aff_R = mask_by_max_delay_window(args, pr, aff_L2R.transpose(-2, -1))
        
        if args.select == 'soft_weight':
            crw_itd_R = torch.sum(masked_aff_R * -delay_time, dim=-1)
            
        elif args.select == 'argmax':
            B, N, M = masked_aff_R.size()
            aff = masked_aff_R.contiguous().view(B * M, -1)
            inds_y = torch.argmax(aff, dim=-1)
            inds_x = torch.arange(aff.shape[0]).to(aff.device)
            crw_itd_R = delay_time.view(B * M, -1)[inds_x, inds_y]
            crw_itd_R = crw_itd_R.view(B, N)

        crw_itd_R = crw_itd_R[:, max_delay_offset: -max_delay_offset]
        crw_itd = torch.cat([crw_itd, crw_itd_R], dim=-1)
    return crw_itd


def inference_crw_itd_with_cycle(args, pr, aff_L2R, delay_time):
    # import pdb; pdb.set_trace()
    max_delay_offset = np.floor(pr.samp_sr * pr.max_delay / pr.patch_stride).astype(int)
    masked_aff = mask_by_max_delay_window(args, pr, aff_L2R)
    masked_aff_R2L = mask_by_max_delay_window(args, pr, aff_L2R.transpose(-2, -1))
    cycle_matrix = torch.matmul(masked_aff, masked_aff_R2L)
    cycle_matrix_R = torch.matmul(masked_aff_R2L, masked_aff)

    N, M, _ = cycle_matrix.shape
    target = torch.eye(M).repeat(N, 1, 1).to(cycle_matrix.device)
    _, labels = target.max(dim=-1)
    _, preds = cycle_matrix.max(dim=-1)
    _, preds_R = cycle_matrix_R.max(dim=-1)


    crw_itd = torch.sum(masked_aff * delay_time, dim=-1)
    crw_itd_R = torch.sum(masked_aff_R2L * -delay_time, dim=-1)

    itds = [] 
    for i in range(N):
        cycle_inds = torch.nonzero(preds[i] == labels[i]).view(-1)
        curr_itd = crw_itd[i, cycle_inds]
        if args.bidirectional:
            cycle_inds_R = torch.nonzero(preds_R[i] == labels[i]).view(-1)
            curr_itd_R = crw_itd_R[i, cycle_inds]
            curr_itd = torch.cat([curr_itd, curr_itd_R], dim=-1)
        itds.append(curr_itd)
    return itds


def ransac_like_pick(args, pr, itds, thres=0.05):
    # import pdb; pdb.set_trace()
    itds = itds * pr.samp_sr
    # thres = 0.05
    min_itd = itds.min()
    max_itd = itds.max()
    poss_itds = torch.linspace(start=min_itd, end=max_itd, steps=1001).to(itds.device)
    counts = []

    distance = torch.abs(poss_itds.view(-1, 1) - itds)
    inliers = distance <= thres
    counts = torch.sum(inliers, dim=-1)
    max_inds = torch.nonzero(counts == counts.max()).view(-1)
    select_ind = max_inds[int(max_inds.shape[0] // 2)]
    select_itds = itds[torch.nonzero(inliers[select_ind]).view(-1)]
    itd = torch.mean(select_itds) / pr.samp_sr
    return itd
    


def estimate_crw_itd(args, pr, aff_L2R, delay_time, no_postprocess=False):
    if args.cycle_filter: 
        crw_itds = inference_crw_itd_with_cycle(args, pr, aff_L2R, delay_time)
    else:
        crw_itds = inference_crw_itd_simple(args, pr, aff_L2R, delay_time)
    
    if no_postprocess: 
        return crw_itds

    itds = []
    for i in range(aff_L2R.shape[0]):
        curr_itds = crw_itds[i]
        if args.mode == 'mean':
            itd = torch.mean(curr_itds)
        elif args.mode == 'ransac':
            itd = ransac_like_pick(args, pr, curr_itds) 
        itds.append(itd)
    itds = torch.stack(itds)
    pool = nn.AdaptiveAvgPool1d(256)
    crw_itds = pool(crw_itds.unsqueeze(1)).squeeze(1)
    # print(crw_itds.shape)

    return itds,crw_itds

# ------------------ Angle2ITD function --------------------------- #

def angle2itd(args, pr, a, b, angle, sound_speed):
    # a = (torch.ones(angle.shape[0]) * a).to(angle.device)
    # b = (torch.ones(angle.shape[0]) * b).to(angle.device)
    angle_1 = (90 + angle) * np.pi / 180
    angle_2 = (90 - angle) * np.pi / 180
    c_1 = cosine_formula(a, b, angle_1)
    c_2 = cosine_formula(a, b, angle_2)
    itd = (c_2 - c_1) / sound_speed
    return itd

def cosine_formula(a, b, angle):
    c_square = a ** 2 + b ** 2 - 2 * a * b * torch.cos(angle)
    c = torch.sqrt(c_square)
    return c

def itd2angle(args, pr, a, b, itd, sound_speed):
    # import pdb; pdb.set_trace()
    angle_sign = 2 * np.ones_like(itd) * (itd <= 0) - 1 
    sin_angle_squared = ((itd * sound_speed)**2 * (a**2 + b**2) * 4 - (itd * sound_speed)**4) / (16 * a**2 * b**2) 
    angle = np.arcsin(np.sqrt(sin_angle_squared))
    angle = angle * angle_sign
    angle = angle * 180 / np.pi
    return angle

# --------------------------------------------------------------- # 


def write_itd(args, pr, pseudo_itds, crw_itds, baseline_itds, gt_angles):
    itd_dict = []
    itd_json = os.path.join('results', args.exp, 'itd.json')
    for i in range(gt_angles.shape[0]):
        temp = {
            'angle': gt_angles[i],
            'math_itd': pseudo_itds[i],
            'crw_itd': crw_itds[i],
            'handcraft_itd': baseline_itds[i]
        }
        itd_dict.append(temp)
    write_csv(itd_dict, itd_json)


def gcc_phat(args, pr, audio, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    audio : (2, N)
    fs: audio rate
    max_tau: max delay
    '''
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    # import pdb; pdb.set_trace()
    # n = int(pr.clip_length * fs)
    # n = 1024
    n = args.gcc_fft
    num = pr.patch_num
    if num == 1:
        step = int(audio.shape[1] - n) + 1
    else:
        step = int((audio.shape[1] - n) // (num - 1))
    audio = audio.unfold(-1, n, step)
    sig = audio[0]
    refsig = audio[1]
    n = 2 * n
    # Generalized Cross Correlation Phase Transform
    SIG = torch.fft.rfft(sig, n=n)
    REFSIG = torch.fft.rfft(refsig, n=n)
    R = SIG * torch.conj(REFSIG)
    cc = torch.fft.irfft(R /(torch.abs(R) + 1e-20), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    cc = torch.cat([cc[:, -max_shift:], cc[:, :max_shift+1]], dim=-1)

    # find max cross correlation index
    shift = torch.argmax(torch.abs(cc), dim=-1) - max_shift
    itd = - shift.float() / float(interp * fs)

    offset = int(fs * pr.max_delay / step)
    if args.same_vote:
        # import pdb; pdb.set_trace()
        itd = itd[offset: -offset]
    if args.mode == 'mean':
        itd = torch.mean(itd)
    elif args.mode == 'ransac':
        itd = ransac_like_pick(args, pr, itd) 
    
    itd = itd.item()
    return itd