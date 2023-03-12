#!/bin/bash

batch_sizes='10000 30000 50000'
learning_rates='0.005 0.01 0.02'

for bs in $batch_sizes; do 
    for lr in $learning_rates; do 
        echo $bs, $lr
        python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
                --discount 0.95 -n 100 -l 2 -s 32 -b $bs -lr $lr -rtg --nn_baseline \
                --exp_name q4_search_b${bs}_lr${lr}_rtg_nnbaseline 
    done
done


