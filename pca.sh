#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --mem=64G

srun ./venv/bin/python3 chest_xray/tools/pca.py
