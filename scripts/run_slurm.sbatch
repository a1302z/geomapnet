#!/bin/bash
#SBATCH --job-name=mapnet-aachendaynight
#SBATCH --nodes=1
#SBATCH --cpus=1
#SBATCH --mem=8GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
srun python train.py --dataset AachenDayNight --model mapnet --config configs/mapnet.ini

