#!/bin/bash

for d in `seq 1 50`
do
    python main2.py $d &
done

wait

python visualize_all_distance.py