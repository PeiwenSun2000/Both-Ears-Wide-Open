#!/bin/bash
{
CUDA=0
Mixture=(0.1 0.3 0.5 0.7 0.9)
batchsize=12

Model='pretrained-models/FreeMusic-StereoCRW-1024.pth.tar' # change to your model

source ~/.bashrc
conda activate Stereo

for (( i=0; i < 5; i++)); 
do {
    echo "Process Mixture=\"${Mixture[$i]}\" started";

    CUDA_VISIBLE_DEVICES=$CUDA nice -n 0 python vis_scripts/eval_itd_by_itd.py --exp='test' --setting='stereocrw_tdesim' --backbone='resnet9' --batch_size=$batchsize --num_workers=4 --resume=$Model --patch_stride=1 --patch_num=128  --clip_length=0.48 --wav2spec --add_sounds=0 --max_weight=${Mixture[$i]} --mode='ransac' --baseline_type='gcc_phat' --noiseSNR=30 --rt60=0.1 --no_baseline --add_mixture

    echo "  ";
    echo "  ";
    echo "  ";
} done


}


