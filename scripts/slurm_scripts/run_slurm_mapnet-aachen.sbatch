#!/bin/bash
#SBATCH --job-name=mapnet-aachendaynight-uncertainty
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model mapnet --config configs/mapnet.ini --server 'atbeetz18.informatik.tu-muenchen.de' --learn_beta --learn_gamma --uncertainty_criterion

