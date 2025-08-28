#!/bin/bash
#SBATCH --job-name=full_eval
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --output=/fsx/arthur_bresnu/projects/sentence-transformers/logs/full_eval/%x-%j.out

cd /fsx/arthur_bresnu/projects/sentence-transformers
source .venv/bin/activate
uv pip install .

MODEL_NAME=models/splade-trans-ettin-encoder-150m-uncased-en-en-msmarco-Qwen3-8B-scores-4-bs_128-lr_8e-05-lq_0.15-ld_0.2/checkpoint-136150
CROP=false  # Set to true if you want to freeze layers
CROP_ARG=""
if [ "$CROP" = true ]; then
    CROP_ARG="--crop"
fi
# Run the evaluation script
python examples/sparse_encoder/evaluation/sparse_nanobeir_advanced_evaluator.py --model_name $MODEL_NAME $CROP_ARG
python examples/sparse_encoder/evaluation/mteb_eval.py --model_name $MODEL_NAME $CROP_ARG
# python examples/sparse_encoder/evaluation/mteb_eval_multilingual.py --model_name $MODEL_NAME $CROP_ARG