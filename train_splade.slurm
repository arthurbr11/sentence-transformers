#!/bin/bash
#SBATCH --job-name=train_splade_marginmse
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:8                 # Request 8 GPUs
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output=/fsx/arthur_bresnu/projects/sentence-transformers/logs/train_splade/%A_%a.out

# Define scripts and GPU assignments
SCRIPTS=(
    "examples/sparse_encoder/training/distillation/train_splade_msmarco_margin_mse.py"
    "examples/sparse_encoder/training/distillation/train_splade_msmarco_margin_mse.py"
)

# Define constants for training parameters
MODEL_NAMES=(
"models/ettin-encoder-68m-lowercase"    "models/ettin-encoder-68m-lowercase"
)
N_GPUS=(4 4)
TRAIN_BATCH_SIZES=(32 32)
NUM_EPOCHS=(37 37)
QUERY_REGULARIZER_WEIGHTS=(0.1 0.08)
DOCUMENT_REGULARIZER_WEIGHTS=(0.1 0.1)
LEARNING_RATES=(3e-4 7e-4)
DATASET_NAMES=("datasets/msmarco-Qwen3-8B-scores-4" "datasets/msmarco-Qwen3-8B-scores-4")

cd /fsx/arthur_bresnu/projects/sentence-transformers
source .venv/bin/activate
uv pip install .


# Run the tasks
for ((TASK_ID=0; TASK_ID<${#SCRIPTS[@]}; TASK_ID++)); do
    SCRIPT=${SCRIPTS[$TASK_ID]}
    MODEL_NAME=${MODEL_NAMES[$TASK_ID]}
    N_GPU=${N_GPUS[$TASK_ID]}
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZES[$TASK_ID]}
    NUM_EPOCH=${NUM_EPOCHS[$TASK_ID]}
    QUERY_REGULARIZER_WEIGHT=${QUERY_REGULARIZER_WEIGHTS[$TASK_ID]}
    DOCUMENT_REGULARIZER_WEIGHT=${DOCUMENT_REGULARIZER_WEIGHTS[$TASK_ID]}
    LEARNING_RATE=${LEARNING_RATES[$TASK_ID]}
    DATASET_NAME=${DATASET_NAMES[$TASK_ID]}

    # Calculate the GPU indices for this task
    GPU_START=$((TASK_ID * 4))
    GPU_END=$((GPU_START + 3))
    export CUDA_VISIBLE_DEVICES=$(seq -s, $GPU_START $GPU_END)

    echo "Task $TASK_ID starting at $(date)"
    echo "Running script: $SCRIPT"
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
    echo "Node: $SLURM_NODELIST"
    echo "Job ID: $SLURM_JOB_ID"

    # Run the script in the background with a random port
    export MASTER_PORT=$((RANDOM % 10000 + 20000))  # Random port between 20000 and 30000
    accelerate launch --num_processes $N_GPU --main_process_port $MASTER_PORT $SCRIPT \
        --model_name "$MODEL_NAME" \
        --n_gpu $N_GPU \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --num_epochs $NUM_EPOCH \
        --query_regularizer_weight $QUERY_REGULARIZER_WEIGHT \
        --document_regularizer_weight $DOCUMENT_REGULARIZER_WEIGHT \
        --learning_rate $LEARNING_RATE \
        --dataset_name $DATASET_NAME > /fsx/arthur_bresnu/projects/sentence-transformers/logs/train_splade/${SLURM_JOB_ID}_${TASK_ID}.out 2>&1 &

done

# Wait for all background tasks to complete
wait

echo "All tasks completed at $(date)"
