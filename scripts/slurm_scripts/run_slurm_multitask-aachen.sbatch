#!/bin/bash
#SBATCH --job-name=multitask-aachendaynight
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model multitask --config_file configs/uncertainty-criterion.ini --server 'atbeetz18.informatik.tu-muenchen.de'

