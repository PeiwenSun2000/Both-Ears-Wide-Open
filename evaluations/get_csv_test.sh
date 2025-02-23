#!/bin/bash

# 设置要扫描的文件夹路径
input_dir="/data/peiwensun/project/sag/infer/mix_cfg_2/"
input_dir="/data/peiwensun/project/stereocrw/fsad_watch"
# 设置输出 CSV 文件名
output_file="fsad_watch.csv"

# 遍历文件夹下所有文件,并将绝对路径写入 CSV 文件
find "$input_dir" -type f | while read file; do
    echo "\"$file\"" >> "$output_file"
done

