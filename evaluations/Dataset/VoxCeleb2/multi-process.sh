#!/bin/bash
read -p '# of subprocess: ' max
frame_rate=5
max_sample=1000


for (( i=0; i < $max; i++)); 
do {
    echo "Process \"$i\" started";
    python process.py --split=$i --total=$max --frame_rate=$frame_rate --max_sample=$max_sample & pid=$!
    PID_LIST+=" $pid";
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";