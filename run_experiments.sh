#!/bin/bash
set -e

for vrandom in 0 1
do
    for epoch in 5 50 100
    do
        ./run_train_self_supervised.sh $vrandom $epoch 200 5
    done
    for bs in 200 1000 5000
    do
        ./run_train_self_supervised.sh $vrandom 5 $bs 5
    done
    for weight in 5 500 5000
    do
        ./run_train_self_supervised.sh $vrandom 5 200 $weight
    done
done
