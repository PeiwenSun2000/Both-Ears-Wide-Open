#!/bin/bash

source ~/.bashrc
conda activate Stereo

EXP='FMA-MonoCLR-0.064S-ResNet9-4Stride-49num'
batch_size=48
num_workers=4
pretrained='pretrained-models/FreeMusic-MonoCLR-7680-training-use.pth.tar' # set to '' if you want to train from scratch 

# running jobs
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp=$EXP --epochs=80 --setting='aug_stereocrw_freemusic' --backbone='resnet9' --batch_size=$batch_size --num_workers=$num_workers --save_step=1 --valid_step=1 --lr=0.0001 --optim='AdamW' --repeat=1 --schedule='cos' --clip_length=0.064 --patch_stride=4 --patch_num=49 --wav2spec --aug_wave --shift_wave --larger_shift --add_mixture --noiseSNR=40 --add_reverb --skip_node --synthetic_rate=1.0 --crw_rate=0.0 --resume=$pretrained