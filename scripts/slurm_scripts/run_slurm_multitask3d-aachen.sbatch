#!/bin/bash
#SBATCH --job-name=multitask3d-aachendaynight
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model multitask3d --config_file configs/uncertainty-criterion_multitask3d.ini --server 'atcremers3.informatik.tu-muenchen.de' --use_augmentation

