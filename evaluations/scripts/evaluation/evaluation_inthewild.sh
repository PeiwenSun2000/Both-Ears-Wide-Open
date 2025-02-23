#!/bin/bash
{
CUDA=0
batchsize=12
Model='pretrained-models/FreeMusic-StereoCRW-1024.pth.tar' # change to your model

source ~/.bashrc
conda activate Stereo

echo "Evaluate In-the-wild data started";

CUDA_VISIBLE_DEVICES=$CUDA nice -n 0 python vis_scripts/eval_itd_in_wild.py \
    --exp='test' \
    --setting='stereocrw_binaural' \
    --backbone='resnet9' \
    --batch_size=$batchsize \
    --num_workers=4 \
    --resume=$Model \
    --patch_size=7680 \
    --patch_stride=1 \
    --patch_num=128  \
    --clip_length=0.48 \
    --wav2spec \
    --mode='ransac' \
    --baseline_type='gcc_phat' \
    --no_baseline
}

