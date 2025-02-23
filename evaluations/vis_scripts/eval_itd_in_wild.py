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


def update_param(args, pr):
    for attr in vars(pr).keys():
        if attr in vars(args).keys():
            attr_args = getattr(args, attr)
            if attr_args is not None:
                setattr(pr, attr, attr_args)


def inference(args, pr, net, criterion, data_loader, device='cuda'):
    # import pdb; pdb.set_trace()
    net.eval()
    itd_labels = []
    crw_itds = []
    baseline_itds = []
    ild_cues = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            audio = batch['audio']
            audio_rate = batch['audio_rate']
            delay_time = batch['delay_time'].to(device)
            itd_label = batch['itd_label'].to(device)
            ild_cue = batch['ild_cue'].to(device)
            out = predict(args, pr, net, batch, device)
            aff_L2R = criterion.inference(out, softmax=False)
            # independent crw itd estimation
            crw_itd = estimate_crw_itd(args, pr, aff_L2R, delay_time)

            crw_itds.append(crw_itd)
            itd_labels.append(itd_label)
            ild_cues.append(ild_cue)
            if args.no_baseline:
                baseline_itd = [0] * aff_L2R.shape[0]
            else:
                baseline_itd = []
                for i in range(aff_L2R.shape[0]):
                    curr_audio = audio[i]
                    if args.baseline_type == 'gcc_phat':
                        hand_itd = gcc_phat(args, pr, curr_audio, fs=audio_rate[i].item(), max_tau=pr.max_delay, interp=1)
                    baseline_itd.append(hand_itd)
            baseline_itds = baseline_itds + baseline_itd
    # import pdb; pdb.set_trace() 
    crw_itds = torch.cat(crw_itds, dim=-1).data.cpu().numpy() * 1000
    baseline_itds = np.array(baseline_itds) * 1000
    itd_labels = torch.cat(itd_labels, dim=-1).data.cpu().numpy()
    ild_cues = torch.cat(ild_cues, dim=-1).data.cpu().numpy()

    # import pdb; pdb.set_trace()
    crw_res = eval_in_wild(crw_itds, itd_labels, 'CRW')
    if args.no_baseline: 
        baseline_res = {}
    else:
        baseline_res = eval_in_wild(baseline_itds, itd_labels, 'Handcraft')
    
    ild_res = eval_in_wild(ild_cues, itd_labels, 'ILD Cue', no_convert=True)

    res = {**crw_res, **baseline_res, **ild_res}
    return res

def convert_itd_to_pred(itd):
    pred = np.zeros_like(itd)
    approx_delay = 1 / 16
    for i in range(itd.shape[0]):
        if itd[i] > approx_delay:
            pred[i] = 0
        elif itd[i] <= approx_delay and itd[i] >= 0:
            pred[i] = 1
        elif itd[i] >= -approx_delay and itd[i] < 0:
            pred[i] = 2
        elif itd[i] < -approx_delay:
            pred[i] = 3
    return pred



def eval_in_wild(pred, target, prefix='', no_convert=False):
    if prefix != '':
        prefix = prefix + ' '
    
    if not no_convert:
        pred = convert_itd_to_pred(pred)
    target[target == 1] = 0
    target[target == 2] = 3

    pred[pred == 1] = 0
    pred[pred == 2] = 3

    acc = (pred == target).sum() / target.shape[0]
    err = (np.abs(pred - target)).mean()
    
    res = {
        f'{prefix}Acc': acc,
        f'{prefix}Error': err,
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
    args = parser.parse_args()
    test(args, DEVICE)