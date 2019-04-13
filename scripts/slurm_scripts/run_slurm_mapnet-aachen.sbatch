#!/bin/bash
#SBATCH --job-name=mapnet-aachendaynight-higher_res
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model mapnet --config configs/mapnet.ini --server 'atbeetz18.informatik.tu-muenchen.de' --crop_size_file crop_size-600x600.txt

