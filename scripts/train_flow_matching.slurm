#!/bin/bash
#SBATCH --job-name=flow_matching
#SBATCH --output=logs/flow_matching_%j.out
#SBATCH --error=logs/flow_matching_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "job started: $(date)"
echo "running on node: $(hostname)"

cd /scratch/zt1/project/msml640psc2/user/yaxita/bitemporal_synthesis

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
source venv/bin/activate

python src/training/train_flow_matching.py

echo "job finished: $(date)"
