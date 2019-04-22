#!/bin/bash
#SBATCH --job-name=multitask-aachendaynight-uncertainty
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model multitask --config_file configs/uncertainty-criterion.ini --server 'atbeetz18.informatik.tu-muenchen.de' --learn_beta --learn_gamma --learn_sigma --uncertainty_criterion

