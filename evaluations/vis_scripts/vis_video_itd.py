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
import data
import models
from models import *
from utils import utils, torch_utils

from vis_functs import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle


def inference(args, pr, net, criterion, data_set, data_loader, device='cuda', video_idx=None):
    net.eval()
    img_path_list = []
    crw_itds = []
    feat_itds = []
    baseline_itds = []
    args.no_baseline = False
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            audio = batch['audio']
            audio_rate = batch['audio_rate']
            delay_time = batch['delay_time'].to(device)
            out = predict(args, pr, net, batch, device)
            aff_L2R = criterion.inference(out, softmax=False)
            crw_itd, feat_itd = estimate_crw_itd(args, pr, aff_L2R, delay_time)
            crw_itds.append(crw_itd)
            feat_itds.append(feat_itd)
            for i in range(aff_L2R.shape[0]):
                curr_audio = audio[i]

                if args.no_baseline:
                    baseline_itd = 0
                else:
                    baseline_itd = gcc_phat(args, pr, curr_audio, fs=audio_rate[i].item(), max_tau=pr.max_delay, interp=1)
                baseline_itds.append(baseline_itd)
    crw_itds = torch.cat(crw_itds, dim=-1).data.cpu().numpy() * 1000
    baseline_itds = np.array(baseline_itds) * 1000
    feat_itds = torch.cat(feat_itds, dim=0)
    return crw_itds, baseline_itds, feat_itds


def test(args, device):
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    fn = getattr(params, args.setting)
    pr = fn()
    pr.dataloader = 'SingleVideoDataset'
    update_param(args, pr)
    sys.stdout = utils.LoggerOutput(os.path.join('results', args.exp, 'log.txt'))
    os.makedirs('./results/' + args.exp, exist_ok=True)
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))

    net = models.__dict__[pr.net](args, pr, device=device).to(device)
    criterion = models.__dict__[pr.loss](args, pr, device)

    if args.resume:
        resume = './checkpoints/' + args.resume
        net, _ = torch_utils.load_model(resume, net, device=device, strict=False)

    if os.path.exists("./temp_1"):
        shutil.rmtree("./temp_1")
    os.makedirs("./temp_1")
    if os.path.exists("./temp_2"):
        shutil.rmtree("./temp_2")
    os.makedirs("./temp_2")

    net = nn.DataParallel(net, device_ids=gpu_ids)
    print("Generating for folder 1")
    pr.list_vis = "temp_1.csv"
    if isinstance(pr.list_vis, str):
        samples = []
        csv_rows = list(csv.DictReader(open(pr.list_vis, 'r'), delimiter=','))
        csv_rows.sort(key=lambda x: x['path'])
        
        i = 0
        for row in csv_rows:
            # i += 1
            # if i > 5:
            #     break
            if not row in samples:
                samples.append(row)

    if args.max_sample > 0:
        samples = samples[: args.max_sample]

    for i in tqdm(range(len(samples)), desc="Generating Video"):
        pr.list_test = samples[i]['path']
        test_dataset, test_loader = torch_utils.get_dataloader(args, pr, split='test', shuffle=False, drop_last=False)
        crw_itds, baseline_itds, feat_itds = inference(args, pr, net, criterion, test_dataset, test_loader, device, video_idx=i)
        with open("./temp_1/"+os.path.basename(pr.list_test)+".pkl", "wb") as f:
            pickle.dump({"crw_itds": crw_itds, "baseline_itds": baseline_itds, "feat_itds": feat_itds}, f)

    print("Generating for folder 2")
    pr.list_vis = "temp_2.csv"
    if isinstance(pr.list_vis, str):
        samples = []
        csv_rows = list(csv.DictReader(open(pr.list_vis, 'r'), delimiter=','))
        csv_rows.sort(key=lambda x: x['path'])
        
        i = 0
        for row in csv_rows:
            # i += 1
            # if i > 5:
            #     break
            if not row in samples:
                samples.append(row)

    if args.max_sample > 0:
        samples = samples[: args.max_sample]
    
    for i in tqdm(range(len(samples)), desc="Generating Video"):
        pr.list_test = samples[i]['path']
        test_dataset, test_loader = torch_utils.get_dataloader(args, pr, split='test', shuffle=False, drop_last=False)
        crw_itds, baseline_itds, feat_itds = inference(args, pr, net, criterion, test_dataset, test_loader, device, video_idx=i)
        with open("./temp_2/"+os.path.basename(pr.list_test)+".pkl", "wb") as f:
            pickle.dump({"crw_itds": crw_itds, "baseline_itds": baseline_itds, "feat_itds": feat_itds}, f)


if __name__ == '__main__':
    parser = init_args(return_parser=True)
    parser.add_argument('--list_vis', type=str, default=None, required=False)
    args = parser.parse_args()
    test(args, DEVICE)