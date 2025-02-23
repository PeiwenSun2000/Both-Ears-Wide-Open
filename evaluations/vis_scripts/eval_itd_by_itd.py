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


def inference(args, pr, net, criterion, data_loader, device='cuda'):
    # import pdb; pdb.set_trace()
    net.eval()
    gt_angles = []
    pseudo_itds = []
    crw_itds = []
    baseline_itds = []
    # crw_itd_diffs = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            # import pdb; pdb.set_trace()
            audio = batch['audio']
            audio_rate = batch['audio_rate']
            delay_time = batch['delay_time'].to(device)
            math_itd = batch['itd'].to(device)
            out = predict(args, pr, net, batch, device)
            aff_L2R = criterion.inference(out, softmax=False)
            # independent crw itd estimation
            crw_itd = estimate_crw_itd(args, pr, aff_L2R, delay_time)

            crw_itds.append(crw_itd)

            pseudo_itds.append(math_itd)


            if args.no_baseline:
                baseline_itd = [0] * aff_L2R.shape[0]
            else:
                baseline_itd = []
                for i in range(aff_L2R.shape[0]):
                    curr_audio = audio[i]
                    if args.baseline_type == 'gcc_phat':
                        hand_itd = gcc_phat(args, pr, curr_audio, fs=audio_rate[i].item(), max_tau=pr.max_delay, interp=1)
                    elif args.baseline_type == 'flipcoin':
                        coin = np.random.random() > 0.5
                        if coin:
                            hand_itd = math_itd[i].item()
                        else:
                            hand_itd = batch['itd_neg'][i].item()
                    baseline_itd.append(hand_itd)
            baseline_itds = baseline_itds + baseline_itd

    crw_itds = torch.cat(crw_itds, dim=-1).data.cpu().numpy() * 1000
    pseudo_itds = torch.cat(pseudo_itds, dim=-1).data.cpu().numpy() * 1000
    baseline_itds = np.array(baseline_itds) * 1000

    crw_res = eval_regression(crw_itds, pseudo_itds, 'CRW')
    if args.no_baseline: 
        baseline_res = {}
    else:
        baseline_res = eval_regression(baseline_itds, pseudo_itds, 'Handcraft')

    res = {**crw_res, **baseline_res}

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


def test(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    update_param(args, pr)
    if args.setting in ['stereocrw_tdesim']:
        pr.list_test = f'data/TDE-Simulation/data-split/test_RT60_{args.rt60}.csv'
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
    parser.add_argument('--rt60', type=float, default=0.3, required=False)

    args = parser.parse_args()
    test(args, DEVICE)