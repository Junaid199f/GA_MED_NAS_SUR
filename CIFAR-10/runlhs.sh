#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpuv100|gpup100
#SBATCH --job-name="NAS-FRAME-CIFAR-10"
#SBATCH -p publicgpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=END         # Réception d'un mail à la fin du job
#SBATCH --mail-user=junaid199f@gmail.com

module load python/Anaconda
source activate gaunet



python3 lhs.py
