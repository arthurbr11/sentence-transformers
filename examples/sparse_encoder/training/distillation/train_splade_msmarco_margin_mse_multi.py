"""
This scripts demonstrates how to train a Sparse Encoder model for Information Retrieval.

As dataset, we use MSMARCO version with hard negatives from the bert-ensemble-margin-mse dataset.

As loss function, we use MarginMSELoss in the SpladeLoss.
"""

import argparse
import logging
import os
import traceback

from datasets import load_from_disk

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.sparse_encoder import evaluation, losses
from sentence_transformers.training_args import MultiDatasetBatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

LANGS = ["en", "de", "es", "fr"]


def main():
    parser = argparse.ArgumentParser(description="Train a Sparse Encoder model for Information Retrieval.")
    parser.add_argument(
        "--model_name", type=str, default="google-bert/bert-base-multilingual-uncased", help="Model name."
    )
    parser.add_argument("--n_gpu", type=int, default=4, help="Number of GPUs to use.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size per device.")
    parser.add_argument("--num_epochs", type=int, default=35, help="Number of training epochs.")
    parser.add_argument("--query_regularizer_weight", type=float, default=0.002, help="Query regularizer weight.")
    parser.add_argument("--document_regularizer_weight", type=float, default=0.04, help="Document regularizer weight.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="datasets/swim-ir-monolingual-Qwen3-8B-scores-4",
        help="Dataset name.",
    )
    os.environ.setdefault("WANDB_PROJECT", "splade-multi-training")

    args = parser.parse_args()
    model_name = args.model_name
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    n_gpu = args.n_gpu
    train_batch_size = args.train_batch_size
    num_epochs = args.num_epochs
    query_regularizer_weight = args.query_regularizer_weight
    document_regularizer_weight = args.document_regularizer_weight
    learning_rate = args.learning_rate
    dataset_name = args.dataset_name
    run_name = f"splade-{short_model_name}-{dataset_name.split('/')[-1]}-bs_{train_batch_size * n_gpu}-lr_{learning_rate}-lq_{query_regularizer_weight}-ld_{document_regularizer_weight}"

    # 1. Define our SparseEncoder model
    model = SparseEncoder(
        model_name,
        model_card_data=SparseEncoderModelCardData(
            model_name=f"models/{run_name}",
        ),
        trust_remote_code=True,  # Set to True if you want to use custom code from the model repository
    )
    model.max_seq_length = 256  # Set the max sequence length to 256 for the training
    # model.tokenizer.do_lower_case = True  # Set to True if the model is lowercase
    # model.tokenizer.add_prefix_space = True  # Add prefix space for models like RoBERTa
    logging.info("Model max length: %s", model.max_seq_length)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
    train_dataset = {}
    eval_dataset = {}

    for lang in LANGS:
        subset_path = os.path.join(dataset_name, lang)
        subset = load_from_disk(subset_path)
        train_dataset[lang] = subset["train"]
        eval_dataset[lang] = subset["eval"]

    logging.info(train_dataset)
    logging.info(eval_dataset)
    # 3. Define our training loss
    loss = losses.SpladeLoss(
        model,
        losses.SparseMarginMSELoss(model),
        query_regularizer_weight=query_regularizer_weight,
        document_regularizer_weight=document_regularizer_weight,
    )

    nano_evaluator = evaluation.SparseNanoBEIREvaluator(batch_size=train_batch_size)
    multilingual_evaluator = evaluation.SparseMtebEvaluator(batch_size=train_batch_size)
    evaluator = SequentialEvaluator(
        evaluators=[nano_evaluator, multilingual_evaluator],
        main_score_function=lambda score: score[-1],
    )
    # 5. Define the training arguments
    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_SparseMTEB_global_average",
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=6800,  # approx. 2 times per epoch
        save_steps=6800,
        save_total_limit=5,
        logging_steps=200,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=42,
    )

    # 6. Create the trainer & start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, using the complete NanoBEIR dataset
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name, private=True)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = SparseEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
