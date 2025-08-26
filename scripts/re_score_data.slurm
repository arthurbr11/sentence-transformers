#!/bin/bash
#SBATCH --job-name=re_score_dataset
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --output=/fsx/arthur_bresnu/projects/sentence-transformers/logs/data/%x-%j.out

cd /fsx/arthur_bresnu/projects/sentence-transformers
source .venv/bin/activate
uv pip install .

# Run the evaluation script
python examples/sparse_encoder/training/distillation/data_creation.py