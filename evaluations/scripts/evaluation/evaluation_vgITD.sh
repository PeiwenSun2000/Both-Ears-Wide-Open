#!/bin/bash
{
CUDA=8
batchsize=4
clip_length=0.96

Model='pretrained-models/VoxCeleb2-AVMonoCLR-15360.pth.tar'  # change to your model

source ~/.bashrc
conda activate Stereo

echo "Evaluate Visually-guided ITD started";

CUDA_VISIBLE_DEVICES=$CUDA nice -n 0 python vis_scripts/eval_voxceleb_itd.py --exp='test' --setting='voxcelebtde_audio_visual_stereocrw' --backbone='resnet9' --batch_size=$batchsize --num_workers=16 --resume=$Model --patch_size=7680 --patch_stride=1 --patch_num=512  --clip_length=$clip_length --wav2spec --mode='ransac' --baseline_type='gcc_phat' --img_feat_scaling=1.0 --eval_classification --crop_face --gcc_fft=15360 --no_baseline 
}
