#!/bin/bash
#SBATCH --job-name=multitask-aachendaynight-gaussian
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model multitask --config_file configs/slurm_tmp_criterions/tmp_gaussian_uncertainty-criterion.ini --server 'atbeetz18.informatik.tu-muenchen.de' --use_augmentation

