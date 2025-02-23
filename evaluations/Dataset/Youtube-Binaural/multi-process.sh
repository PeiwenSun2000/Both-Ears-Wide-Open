#!/bin/bash
read -p '# of subprocess: ' max
dataset_name='Youtube-RacingCar'
out_dataset_name='Youtube-RacingCar-30FPS'
frame_rate=30


for (( i=0; i < $max; i++)); 
do {
    echo "Process \"$i\" started";
    python process.py --split=$i --total=$max --dataset_name=$dataset_name --out_dataset_name=$out_dataset_name --frame_rate=$frame_rate & pid=$!
    PID_LIST+=" $pid";
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";