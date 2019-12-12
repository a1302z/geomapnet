#!/bin/bash
#SBATCH --job-name=localisation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --time=300:00:00
#SBATCH --gres=gpu:1,VRAM:12GB
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model mapnet --config configs/mapnet.ini --server 'atbeetz21.informatik.tu-muenchen.de' --learn_beta --learn_gamma --port 8834 --use_synthetic --use_stylization 16
