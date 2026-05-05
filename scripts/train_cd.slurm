#!/bin/bash
#SBATCH --job-name=train_cd
#SBATCH --output=logs/train_cd_%j.out
#SBATCH --error=logs/train_cd_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=msml640psc2-class

cd /scratch/zt1/project/msml640psc2/user/yaxita/bitemporal_synthesis
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cudnn
source venv/bin/activate

python src/training/train_change_detection.py

echo "all configs done!"
