#!/bin/bash
#SBATCH --time=20:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=8
module purge
module load CUDA-Python/12.6.2.post1-gfbf-2024a-CUDA-12.6.0
source venv/bin/activate
export WANDB_API_KEY=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
srun --nodes=1 --gpus-per-node=1 --time=20:00 --mem=32G --cpus-per-gpu=8 python3 -m main $1
