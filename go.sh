#!/bin/bash

for d in `seq 1 99`
do
    python main.py $d &
done

wait

python visualize_all_distance.py