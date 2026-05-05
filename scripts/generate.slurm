#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=msml640psc2-class

cd /scratch/zt1/project/msml640psc2/user/yaxita/bitemporal_synthesis
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cudnn
source venv/bin/activate

python src/generation/generate_pairs.py

echo "generation done!"
