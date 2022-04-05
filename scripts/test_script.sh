#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 0:05:00

python3 code/training.py