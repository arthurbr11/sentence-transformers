#!/bin/bash
#SBATCH --job-name=train_splade_margin_mse_french
#SBATCH --partition=hopper-extra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --qos=high
#SBATCH --requeue
#SBATCH --output=/fsx/arthur_bresnu/projects/sentence-transformers/logs/train_splade_french/%x-%j.out

cd /fsx/arthur_bresnu/projects/sentence-transformers
source .venv/bin/activate
uv pip install .

# Run the evaluation script
accelerate launch --num_processes 8 examples/sparse_encoder/training/distillation/train_splade_msmarco_margin_mse_french.py