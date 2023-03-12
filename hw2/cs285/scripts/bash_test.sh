#!/bin/bash

batch_sizes='10000 30000 50000'
learning_rates='0.005, 0.01, 0.02'

for bs in $batch_sizes; do 
    for lr in $learning_rates; do 
        echo test${bs}_hello${lr}_bye
    done
done
