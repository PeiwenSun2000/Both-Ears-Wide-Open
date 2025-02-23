#!/bin/bash

source ~/.bashrc
conda activate Stereo

EXP='VoxCeleb2-AVMonoCLR-0.96S-ResNet9-4Stride-49num'
batch_size=48
num_workers=4

# running jobs
CUDA_VISIBLE_DEVICES=0 python main.py --exp=$EXP --epochs=400 --setting='voxceleb_audio_visual_augstereocrw' --backbone='resnet9' --batch_size=$batch_size --num_workers=$num_workers --save_step=5 --valid_step=5 --lr=0.0001 --optim='AdamW' --repeat=1 --schedule='cos' --clip_length=0.96 --patch_stride=4 --patch_num=49 --wav2spec --aug_wave --regular_scaling --synthetic_rate=1.0 --crw_rate=0.0 --img_feat_scaling=1.0 --crop_face --ignore_speaker