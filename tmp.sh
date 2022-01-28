#!/bin/bash
#SBATCH -N 4
#SBATCH -A Anak_%j
#SBATCH -p longq7-mri
#SBATCH --gres=gpu:v100:4
#SBATCH -e tmp_%j.err
#SBATCH -o tmp_%j.out

set -e

# ./run_train_self_supervised.sh 0 5 200 5

# batch size experiment
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 10000
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 50000

# epoch experiment
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 100 --bs 1000

# weight experiment
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000 --max_random_weight_range 5 --use_random_weight_to_benchmark_ef_iwf
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000 --max_random_weight_range 500 --use_random_weight_to_benchmark_ef_iwf
# /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000 --max_random_weight_range 5000 --use_random_weight_to_benchmark_ef_iwf

python3 train_self_supervised.py -d mooc_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000 --ws_multiplier 1 --custom_prefix tmp --ws_framework forward --use_ef_iwf_weight --edge_weight_multiplier 0.1

python3 train_self_supervised.py -d mooc_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000 --ws_multiplier 1 --custom_prefix tmp --ws_framework forward --use_ef_iwf_weight --edge_weight_multiplier 50

python3 train_self_supervised.py -d mooc_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000 --ws_multiplier 1 --custom_prefix tmp --ws_framework forward --use_ef_iwf_weight --edge_weight_multiplier 500
