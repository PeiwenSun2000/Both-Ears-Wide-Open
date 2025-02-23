#!/bin/bash
frame_rate=10

echo "Name of the video folder: $1"
echo "Frame rate: $frame_rate"

python process.py --split=0 --total=1 --dataset_name=$1 --frame_rate=$frame_rate 
python create-csv.py --dataset_name=$1 --type='' --data_split='1:0:0' --unshuffle 

