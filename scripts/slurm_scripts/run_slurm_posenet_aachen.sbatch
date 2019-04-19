#!/bin/bash
#SBATCH --job-name=posenet-aachen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --model posenet --dataset AachenDayNight --config configs/posenet.ini --server atbeetz18.informatik.tu-muenchen.de
