#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48GB

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source venv/bin/activate
srun python3 main.py
