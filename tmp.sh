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

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 200  --ws_framework ensemble --custom_prefix tmp --ws_multiplier 1

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 200  --ws_framework ensemble --custom_prefix tmp --ws_multiplier 1 --init_n_instances_as_multiple_of_ws 5 --fix_begin_data_ind_of_models_in_ensemble

python train_self_supervised.py -d reddit_100000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework ensemble --custom_prefix tmp --ws_multiplier 1 --init_n_instances_as_multiple_of_ws 5 --fix_begin_data_ind_of_models_in_ensemble

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1 --use_nf_iwf_weight

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 2 --window_stride_multiplier 1 --use_nf_iwf_weight

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 3 --window_stride_multiplier 1 --use_nf_iwf_weight
