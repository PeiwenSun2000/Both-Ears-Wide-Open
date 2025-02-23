#!/bin/bash
read -p '# of subprocess: ' max

for (( i=0; i < $max; i++)); 
do {
    echo "Process \"$i\" started";
    python download.py --split=$i --total=$max & pid=$!
    PID_LIST+=" $pid";
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";