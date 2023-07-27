#!/bin/bash

for d in `seq 1 50`
do
    python main.py $d &
done

wait

python visualize_misalignment.py