#!/bin/bash

#SBATCH --account=mscagpu
#SBATCH --job-name=satellite
#SBATCH --output=%j_out.log
#SBATCH --error=%j_err.log
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mscagpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20000


# Load modules:
module load Anaconda3/5.3.0 cuda/9.0 #python/3.5.2

#pip install --user keras-rl
pip install --user gym[atari]
pip install --user h5py
pip install --user Pillow

# Activate conda environment:
#source activate z11

# GPU-based executable:
python3 sent_yuri_noisy.py
