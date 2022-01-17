#!/bin/bash
#SBATCH -N 1
#SBATCH -A Anak_%j
#SBATCH -p longq7
#SBATCH --gres=gpu:v100:4
#SBATCH -e run_with_slurm_%j.err
#SBATCH -o run_with_slurm_%j.out
#SBATCH --mem-MaxMemPerNode

# SBATCH -p longq7-mri
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

module load slurm
module load miniconda3-4.6.14-gcc-8.3.0-eenl5dj

module load cuda-10.1.243-gcc-8.3.0-ti55azn
module load openblas-0.3.7-gcc-8.3.0-oqk2bly
module load fftw-3.3.8-gcc-8.3.0-wngh6wh
module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt

sbatch run_experiments.sh
