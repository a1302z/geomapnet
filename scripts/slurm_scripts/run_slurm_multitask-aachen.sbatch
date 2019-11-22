#!/bin/bash
#SBATCH --job-name=localisation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1,VRAM:12GB
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model multitask --config_file configs/multitask.ini --server 'atbeetz21.informatik.tu-muenchen.de' --port 8834 --learn_beta --learn_gamma --learn_sigma --use_stylization
