#!/bin/bash
{
CUDA=0
batchsize=12
rt60=(0.1 0.3 0.5 0.7 0.9)
snr=(30 20 10 0 -10)

Model='pretrained-models/FreeMusic-StereoCRW-1024.pth.tar' # change to your model

source ~/.bashrc
conda activate Stereo

for (( i=0; i < 5; i++)); 
do {
    for (( j=0; j < 5 ; j++)); 
    do {
        echo "Condition: SNR=\"${snr[$j]}\" and RT60=\"${rt60[$i]}\"";

        CUDA_VISIBLE_DEVICES=$CUDA nice -n 1 python vis_scripts/eval_itd_by_itd.py --exp='test' --setting='stereocrw_tdesim' --backbone='resnet9' --batch_size=$batchsize --num_workers=4 --resume=$Model --patch_size=1024 --patch_stride=1 --patch_num=128  --clip_length=0.064 --wav2spec --add_sounds=0 --max_weight=0.0  --mode='ransac' --baseline_type='gcc_phat' --noiseSNR=${snr[$j]} --rt60=${rt60[$i]} --no_baseline

        echo "  ";
        echo "  ";
        echo "  ";
    } done
} done


}

