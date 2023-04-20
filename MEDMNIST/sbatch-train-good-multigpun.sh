#! /bin/bash

#SBATCH -p publicgpu          # Partition public avec des GPU
#SBATCH -N 3                  # 1 nœud
#SBATCH --exclusive           # Le nœud sera entièrement dédié à notre job, pas de partage de ressources
#SBATCH -t 24:00:00           # Le job sera tué au bout de 5h
#SBATCH --gres=gpu:3          # 4 GPU par nœud
#SBATCH --constraint=gpudp    # Nœuds GPU double précision
#SBATCH --mail-type=END       # Réception d'un mail à la fin du job
#SBATCH --mail-user=junaid199f@gmail.com

module load gcc/gcc-10

module load python/Anaconda
source activate base


python3 mainn.py
