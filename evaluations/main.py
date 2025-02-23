import argparse
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import init_args, params
# from data import *
import data
import models
from models import *
from utils import utils, torch_utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_param(args, pr):
    for attr in vars(pr).keys():
        if attr in vars(args).keys():
            attr_args = getattr(args, attr)
            if attr_args is not None:
                setattr(pr, attr, attr_args)
    if args.crop_face:
        pr.img_size = 160
        pr.crop_size = 160


def validation(args, pr, net, criterion, data_loader, device='cuda'):
    net.eval()
    res = {}
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation"):
            out = predict(args, pr, net, batch, device, evaluate=True)
            for key in out.keys():
                if key not in res.keys():
                    res[key] = torch.tensor([]).to(device)
                res[key] = torch.cat([res[key], out[key].view(1, -1)], dim=0)

    for key in res.keys():
        res[key] = torch.mean(res[key]).item()
    torch.cuda.empty_cache()
    net.train()
    return res


def predict(args, pr, net, batch, device, evaluate=False, loss=False):
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


def train(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    update_param(args, pr)
    # ----- make dirs for checkpoints ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('checkpoints', args.exp, 'log.txt'))
    os.makedirs('./checkpoints/' + args.exp, exist_ok=True)

    writer = SummaryWriter(os.path.join('./checkpoints', args.exp, 'visualization'))
    # ------------------------------------- #
    
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #

    # ----- Dataset and Dataloader ----- #
    train_dataset, train_loader = torch_utils.get_dataloader(args, pr, split='train', shuffle=True, drop_last=True)
    val_dataset, val_loader = torch_utils.get_dataloader(args, pr, split='val', shuffle=False, drop_last=True)
    # --------------------------------- #
    
    # ----- Network ----- #
    net = models.__dict__[pr.net](args, pr, device=device).to(device)
    criterion = models.__dict__[pr.loss](args, pr, device).to(device)
    optimizer = torch_utils.make_optimizer(net, args)
    # --------------------- #

    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = './checkpoints/' + args.resume
        net, args.start_epoch = torch_utils.load_model(resume, net, device=device, strict=False)
        if args.resume_optim:
            tqdm.write('loading optimizer...')
            optim_state = torch.load(resume)['optimizer']
            optimizer.load_state_dict(optim_state)
            tqdm.write('loaded optimizer!')
        else:
            args.start_epoch = 0

    # ------------------- 
    net = nn.DataParallel(net, device_ids=gpu_ids)

    #  --------- Random or resume validation ------------ #
    res = validation(args, pr, net, criterion, val_loader, device)
    writer.add_scalars('StereoCRW' + '/validation', res, args.start_epoch)
    tqdm.write("Beginning, Validation results: {}".format(res))
    tqdm.write('\n')

    # ----------------- Training ---------------- #
    # import pdb; pdb.set_trace()
    net.train()
    VALID_STEP = args.valid_step
    for epoch in range(args.start_epoch, args.epochs):
        running_loss = 0.0
        # net.train()
        torch_utils.adjust_learning_rate(optimizer, epoch, args, pr)
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            out = predict(args, pr, net, batch, device, loss=True)
            loss = out.mean()      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                tqdm.write("Epoch: {}/{}, step: {}/{}, loss: {}".format(epoch+1, args.epochs, step+1, len(train_loader), loss))
                running_loss += loss.item()

            current_step = epoch * len(train_loader) + step + 1
            BOARD_STEP = 3
            if (step+1) % BOARD_STEP == 0:
                writer.add_scalar('StereoCRW' + '/training loss', running_loss / BOARD_STEP, current_step)
                running_loss = 0.0
            
            if (current_step + 1) % VALID_STEP == 0 and args.valid_by_step:
                res = validation(args, pr, net, criterion, val_loader, device)
                writer.add_scalars('StereoCRW' + '/validation-on-Step', res, current_step)
                tqdm.write("Step: {}/{}, Validation results: {}".format(current_step , args.epochs * len(train_loader), res))
                # tqdm.write('\n')
            
        # torch.cuda.empty_cache()
        # ----------- Validtion -------------- #
        if (epoch + 1) % VALID_STEP == 0 and not args.valid_by_step:
            res = validation(args, pr, net, criterion, val_loader, device)
            writer.add_scalars('StereoCRW' + '/validation', res, epoch + 1)
            tqdm.write("Epoch: {}/{}, Validation results: {}".format(epoch + 1, args.epochs, res))
            # tqdm.write('\n')

        # ---------- Save model ----------- #
        SAVE_STEP = args.save_step
        if (epoch + 1) % SAVE_STEP == 0:
            path = os.path.join('./checkpoints', args.exp, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar')
            torch.save({'epoch': epoch + 1,
                        'step': current_step,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                        path)
        # --------------------------------- #
    torch.cuda.empty_cache()
    tqdm.write('Training Complete!')
    writer.close()


if __name__ == '__main__':
    args = init_args()
    train(args, DEVICE)