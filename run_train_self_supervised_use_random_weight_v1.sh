#!/bin/bash

#SBATCH -N 1
#SBATCH -A Anak_%j
#SBATCH -p longq7
#SBATCH --gres=gpu:v100:4
#SBATCH -e run_train_self_supervised_%j.err
#SBATCH -o run_train_self_supervised_%j.out
#SBATCH --mem-MaxMemPerNode

set -e

nepoch=$1
bs=$2
mweight=$3

if [[ $HOME == "/home/awannaphasch2016" ]]; then
    python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch $nepoch --bs $bs --max_random_weight_range $mweight --use_random_weight_to_benchmark_ef_iwf_1

elif [[ $HOME == "/mnt/beegfs/home/awannaphasch2016" ]]; then

    /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch $nepoch --bs $bs --max_random_weight_range $mweight --use_random_weight_to_benchmark_ef_iwf_1

else
    echo "NotImplementError"
fi
