#!/bin/bash
#SBATCH --job-name=mapnet-aachendaynight-lr1e-4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model mapnet --config configs/mapnet.ini --server 'atbeetz18.informatik.tu-muenchen.de'

