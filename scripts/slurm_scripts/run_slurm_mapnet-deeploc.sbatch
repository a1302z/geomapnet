#!/bin/bash
#SBATCH --job-name=mapnet-deeploc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
srun python train.py --dataset DeepLoc --model mapnet --config configs/mapnet.ini --server 'atcremers52.informatik.tu-muenchen.de'

