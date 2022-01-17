#!/bin/bash
#SBATCH -N 1
#SBATCH -A Anak_%j
#SBATCH -p longq7-mri
#SBATCH --gres=gpu:v100:4
#SBATCH -e run_train_self_supervised_%j.err
#SBATCH -o run_train_self_supervised_%j.out
#SBATCH --mem-MaxMemPerNode

# SRUN -N 1
# SRUN -A Anak_%j
# SRUN -p longq7-mri
# SRUN --gres=gpu:v100:4
# SRUN -e run_train_self_supervised_%j.err
# SRUN -o run_train_self_supervised_%j.out
# SRUN --mem-MaxMemPerNode


# # SBATCH -p shortq7
# # SBATCH --mail-user=awannaphasch2016@fau.edu
# # SBATCH --mail-type=ALL

set -e

nepoch=$1
bs=$2
mweight=$3

if [[ $HOME == "/home/awannaphasch2016" ]]; then
    python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch $nepoch --bs $bs --max_random_weight_range $mweight --use_random_weight_to_benchmark_ef_iwf

elif [[ $HOME == "/mnt/beegfs/home/awannaphasch2016" ]]; then

    /mnt/beegfs/home/awannaphasch2016/.conda/envs/py38/bin/python3 train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch $nepoch --bs $bs --max_random_weight_range $mweight --use_random_weight_to_benchmark_ef_iwf

else
    echo "NotImplementError"
fi
