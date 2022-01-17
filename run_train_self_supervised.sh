#!/bin/bash

set -e

random=$1

if [[ $random == 0 ]]; then
    ./run_train_self_supervised_use_random_weight_v0.sh $2 $3 $4
elif [[ $random == 1 ]]; then
    ./run_train_self_supervised_use_random_weight_v1.sh $2 $3 $4
else
    echo "please specific version of random weight method."
fi
