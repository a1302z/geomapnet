#!/bin/bash
#SBATCH --job-name=posenet-cambridge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --model posenet --dataset CambridgeLandmarks --scene KingsCollege --config configs/posenet.ini --server atcremers3.informatik.tu-muenchen.de
