#!/bin/bash
max=10

dataset_name='Youtube-IntheWild'
data_csv='in-the-wild'
frame_rate=30

for (( i=0; i < $max; i++)); 
do {
    echo "Process \"$i\" started";
    python download.py --split=$i --total=$max --dataset_name=$dataset_name --data_csv=$data_csv & pid=$!
    PID_LIST+=" $pid";
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST


echo
echo "All processes have completed";

for (( i=0; i < $max; i++)); 
do {
    echo "Process \"$i\" started";
    python process.py --split=$i --total=$max --dataset_name=$dataset_name --frame_rate=$frame_rate & pid=$!
    PID_LIST+=" $pid";
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";

