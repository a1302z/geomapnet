#!/bin/bash
#SBATCH --job-name=mapnet-aachendaynight
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mail-type=ALL
srun python train.py --dataset AachenDayNight --model mapnet --config configs/mapnet.ini --server 'atbeetz21.informatik.tu-muenchen.de'

