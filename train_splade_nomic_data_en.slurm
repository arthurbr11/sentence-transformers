#!/bin/bash
#SBATCH --job-name=train_splade_nomic_data_en
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
#SBATCH --output=/fsx/arthur_bresnu/projects/sentence-transformers/logs/train_splade_nomic_data_en/%x-%j.out

cd /fsx/arthur_bresnu/projects/sentence-transformers
source .venv/bin/activate
uv pip install .

SCRIPT="examples/sparse_encoder/training/retrievers/train_splade_nomic_data_en_mnrl.py"

MODEL_NAME="models/ettin-encoder-150m-lowercase"
N_GPU=8
TRAIN_BATCH_SIZE=48
NUM_EPOCH=25
QUERY_REGULARIZER_WEIGHT=8e-4
DOCUMENT_REGULARIZER_WEIGHT=1e-3
LEARNING_RATE=7e-5
DATASET_NAME="datasets/nomic-4"

# Run the evaluation script
accelerate launch --num_processes $N_GPU $SCRIPT \
    --model_name "$MODEL_NAME" \
    --n_gpu $N_GPU \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --num_epochs $NUM_EPOCH \
    --query_regularizer_weight $QUERY_REGULARIZER_WEIGHT \
    --document_regularizer_weight $DOCUMENT_REGULARIZER_WEIGHT \
    --learning_rate $LEARNING_RATE \
    --dataset_name $DATASET_NAME 