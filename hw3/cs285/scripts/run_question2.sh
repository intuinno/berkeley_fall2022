#!/bin/bash

seeds='1 2 3 4 5 6 7 8 9 10'
for seed in $seeds; do 
    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_${seed} --seed $seed &  
    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_${seed} --seed $seed &
done


