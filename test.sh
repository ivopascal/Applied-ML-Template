#!/bin/bash

#SBATCH --time=01:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

#SBATCH --job-name="train $1"

module purge
module load CUDA-Python/12.6.2.post1-gfbf-2024a-CUDA-12.6.0
source /scratch/s3668320/AML/venv/bin/activate

srun --time=01:00 --nodes=1 --partition=gpu --mem=1G --cpus-per-task=1 --gpus-per-node=1 python3 test.py
