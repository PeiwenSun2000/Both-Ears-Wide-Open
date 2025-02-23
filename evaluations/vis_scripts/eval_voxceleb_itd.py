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

from vis_functs import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(args, pr, net, batch, device, evaluate=False, loss=False, speaker2=False):
    # import pdb; pdb.set_trace()
    inputs = {}
    inputs['left_audios'] = batch['left_audios'].to(device)
    inputs['right_audios'] = batch['right_audios'].to(device)
    inputs['delay_time'] = batch['delay_time'].to(device)
    
    if args.setting.find('aug') != -1:
        inputs['noaug_left_audios'] = batch['noaug_left_audios'].to(device)
        inputs['shift_offset'] = batch['shift_offset'].to(device)

    if args.setting.find('audio_visual') != -1:
        inputs['img'] = batch['neg_img'].to(device) if speaker2 else batch['img'].to(device)
        inputs['shift_offset'] = batch['shift_offset'].to(device)
    
    out = net(inputs, evaluate=evaluate, loss=loss)
    return out

def inference(args, pr, net, criterion, data_loader, device='cuda'):
    # import pdb; pdb.set_trace()
    net.eval()
    gt_angles = []
    pseudo_itds = []
    crw_itds = []
    baseline_itds = []
    crw_itd_diffs = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            audio = batch['audio'].to(device)
            audio_rate = batch['audio_rate']
            delay_time = batch['delay_time'].to(device)
            speaker1_itd = batch['itd'].to(device)
            speaker2_itd = batch['itd_neg'].to(device)
            if 'separated_speaker1_audio' in batch.keys():
                separated_speaker1_audio = batch['separated_speaker1_audio'].to(device)
            if 'separated_speaker2_audio' in batch.keys():
                separated_speaker2_audio = batch['separated_speaker2_audio'].to(device)

            out = predict(args, pr, net, batch, device, speaker2=False)
            aff_L2R = criterion.inference(out, softmax=False)
            speaker1_crw_itd = estimate_crw_itd(args, pr, aff_L2R, delay_time)

            out = predict(args, pr, net, batch, device, speaker2=True)
            aff_L2R = criterion.inference(out, softmax=False)
            speaker2_crw_itd = estimate_crw_itd(args, pr, aff_L2R, delay_time)

            crw_itds.append(speaker1_crw_itd)
            crw_itds.append(speaker2_crw_itd)

            pseudo_itds.append(speaker1_itd)
            pseudo_itds.append(speaker2_itd)

            if args.no_baseline:
                speaker1_baseline_itd = [0] * aff_L2R.shape[0]
                speaker2_baseline_itd = [0] * aff_L2R.shape[0]

            else:
                speaker1_baseline_itd = []
                speaker2_baseline_itd = []

                for i in range(aff_L2R.shape[0]):
                    curr_audio = audio[i]
                    if args.baseline_type == 'gcc_phat':
                        hand_itd = gcc_phat(args, pr, curr_audio, fs=audio_rate[i].item(), max_tau=pr.max_delay, interp=1)
                        speaker1_baseline_itd.append(hand_itd)
                        speaker2_baseline_itd.append(hand_itd)

                    elif args.baseline_type == 'flipcoin':
                        coins = (np.random.random(2) > 0.5).astype(int)
                        pool = np.array([speaker1_itd[i].item(), speaker2_itd[i].item()])
                        hand_itds = pool[coins]
                        speaker1_baseline_itd.append(hand_itds[0])
                        speaker2_baseline_itd.append(hand_itds[1])
                    
            baseline_itds = baseline_itds + speaker1_baseline_itd + speaker2_baseline_itd

    crw_itds = torch.cat(crw_itds, dim=-1).data.cpu().numpy() * 1000
    pseudo_itds = torch.cat(pseudo_itds, dim=-1).data.cpu().numpy() * 1000
    baseline_itds = np.array(baseline_itds) * 1000

    crw_res = eval_regression(crw_itds, pseudo_itds, 'CRW')
    if args.no_baseline: 
        baseline_res = {}
    else:
        baseline_res = eval_regression(baseline_itds, pseudo_itds, 'Handcraft')
    
    res = {**crw_res, **baseline_res}

    if args.eval_classification:
        crw_class_res = eval_regression_by_classification(crw_itds, pseudo_itds, prefix='CRW')
        if args.no_baseline: 
            baseline_class_res = {}
        else:
            baseline_class_res = eval_regression_by_classification(baseline_itds, pseudo_itds, 'Handcraft')
        res = {**res, **crw_class_res, **baseline_class_res}
    return res


def eval_regression(pred, target, prefix=''):
    if prefix != '':
        prefix = prefix + ' '
    errs = np.abs(pred - target)
    rmse = np.sqrt(np.mean(errs ** 2))
    mean_err = np.mean(errs)
    median_err = np.median(errs)
    r2 = r2_score(target, pred)
    res = {
        f'{prefix}Mean Error': mean_err,
        f'{prefix}Median Error': median_err,
        f'{prefix}RMSE': rmse,
        f'{prefix}R2 Score': r2
    }
    return res


def eval_regression_by_classification(pred, target, prefix=''):
    if prefix != '':
        prefix = prefix + ' '
    errs = np.abs(pred - target)
    class_1 = (errs >= 0) & (errs <= 0.1)
    class_2 = (errs > 0.1) & (errs <= 0.2)
    class_3 = (errs > 0.2) & (errs <= 0.3)
    class_4 = (errs > 0.3)

    class_1_acc = class_1.sum() / class_1.shape[0]
    class_2_acc = class_2.sum() / class_2.shape[0]
    class_3_acc = class_3.sum() / class_3.shape[0]
    class_4_acc = class_4.sum() / class_4.shape[0]
    
    res = {
        f'{prefix}class 1 acc': class_1_acc,
        f'{prefix}class 2 acc': class_2_acc,
        f'{prefix}class 3 acc': class_3_acc,
        f'{prefix}class 4 acc': class_4_acc,
    }


    return res


def test(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    update_param(args, pr)

    if args.sim_setup == 'Easy':
        pr.list_test = f'data/VoxCeleb2/data-split/voxceleb-tde/Easy/test.csv'
    elif args.sim_setup == 'Hard':
        pr.list_test = f'data/VoxCeleb2/data-split/voxceleb-tde/Hard/test.csv'

    # ----- make dirs for results ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('results', args.exp, 'log.txt'))
    os.makedirs('./results/' + args.exp, exist_ok=True)
    # ------------------------------------- #
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #
    
    # ----- Dataset and Dataloader ----- #
    test_dataset, test_loader = torch_utils.get_dataloader(args, pr, split='test', shuffle=False, drop_last=False)
    # --------------------------------- #

    # ----- Network ----- #
    net = models.__dict__[pr.net](args, pr, device=device).to(device)
    criterion = models.__dict__[pr.loss](args, pr, device)

    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = './checkpoints/' + args.resume
        net, _ = torch_utils.load_model(resume, net, device=device, strict=False)

    # ------------------- #
    net = nn.DataParallel(net, device_ids=gpu_ids)

    #  --------- Testing ------------ #
    res = inference(args, pr, net, criterion, test_loader, device)
    tqdm.write("Testing results: {}".format(res))


if __name__ == '__main__':
    parser = init_args(return_parser=True)
    parser.add_argument('--eval_classification', default=False, action='store_true', help='Visualize itd prediction if True.')
    parser.add_argument('--sim_setup', type=str, default='Easy', required=False)

    args = parser.parse_args()
    test(args, DEVICE)