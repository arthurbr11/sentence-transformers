#!/bin/bash
#SBATCH --job-name=train_mlm
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --output=/fsx/arthur_bresnu/projects/sentence-transformers/logs/train_mlm/%x-%j.out

cd /fsx/arthur_bresnu/projects/sentence-transformers
source .venv/bin/activate
uv pip install .

# Global variables for all script arguments
MODEL="sparse-encoder/trans-ettin-encoder-150m-uncased-en-en"
HF_DATASET="sentence-transformers/msmarco-corpus"
HF_SUBSET="passage"
HF_SPLIT="train"
HF_TEXT_FIELD="text"
MAX_LENGTH=256
MLM_PROBABILITY=0.3
EPOCHS=1
BATCH_SIZE=128
LEARNING_RATE=8e-4
WARMUP_RATIO=0.0
WEIGHT_DECAY=1e-5
SEED=42
BF16=true
LAMBDA_FLOPS=1e-3
FLOPS_POOLING="max"
LOGGING_STEPS=50
SAVE_STEPS=500
FREEZE=false  # Set to true if you want to freeze layers

# Calculate global batch size
NUM_GPUS=8
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Extract model name for naming
MODEL_NAME=$(basename "$MODEL")
FREEZE_STR=$([ "$FREEZE" = true ] && echo "frozen" || echo "unfrozen")

# Dynamic naming based on your requirements
OUTPUT_DIR="outputs/mlm_train_${MODEL_NAME}_gbs${GLOBAL_BATCH_SIZE}_lr${LEARNING_RATE}_${FREEZE_STR}"
WANDB_PROJECT="mlm-train"
WANDB_RUN_NAME="mlm_train_${MODEL_NAME}_gbs${GLOBAL_BATCH_SIZE}_lr${LEARNING_RATE}_${FREEZE_STR}"

echo "MLM Training starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE (${BATCH_SIZE} per GPU × ${NUM_GPUS} GPUs)"
echo "Learning Rate: $LEARNING_RATE"
echo "Freeze: $FREEZE_STR"
echo "Output directory: $OUTPUT_DIR"
echo "W&B Run Name: $WANDB_RUN_NAME"

# Build command with freeze option if needed
FREEZE_ARG=""
if [ "$FREEZE" = true ]; then
    FREEZE_ARG="--freeze"
fi

BF16_ARG=""
if [ "$BF16" = true ]; then
    BF16_ARG="--bf16"
fi

# Run the MLM training script
accelerate launch --num_processes $NUM_GPUS  mlm/train.py \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --hf_dataset "$HF_DATASET" \
    --hf_subset "$HF_SUBSET" \
    --hf_split "$HF_SPLIT" \
    --hf_text_field "$HF_TEXT_FIELD" \
    --max_length $MAX_LENGTH \
    --mlm_probability $MLM_PROBABILITY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    $BF16_ARG \
    --lambda_flops $LAMBDA_FLOPS \
    --flops_pooling "$FLOPS_POOLING" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    $FREEZE_ARG

echo "MLM Training completed at $(date)"