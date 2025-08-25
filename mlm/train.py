from __future__ import annotations

import argparse
import logging
import os

from data import HFDatasetText, MultilingualDataset, SimpleMlmc, TextFileDataset
from trainer import MergeAndRoundLogsCallback, MlmTrainer
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="MLM training with FLOPS regularizer and WandB logging")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for AutoModelForMaskedLM")
    parser.add_argument("--train_file", type=str, default=None, help="Path to a text file with one example per line")
    parser.add_argument("--output_dir", type=str, required=True)

    # HF dataset (optional)
    parser.add_argument(
        "--hf_dataset", type=str, default=None, help="HF dataset name like 'sentence-transformers/msmarco-corpus'"
    )
    parser.add_argument("--hf_subset", type=str, default=None, help="HF dataset subset like 'passage' or 'query'")
    parser.add_argument("--hf_split", type=str, default="train", help="HF dataset split, e.g., 'train'")
    parser.add_argument("--hf_text_field", type=str, default="text", help="Field name containing the text")

    # Multilingual datasets
    parser.add_argument(
        "--multilingual_datasets",
        type=str,
        nargs="*",
        default=None,
        help="List of dataset paths/directories for multilingual training",
    )
    parser.add_argument(
        "--balance_languages", action="store_true", help="Balance datasets to have equal samples per language"
    )
    parser.add_argument(
        "--max_samples_per_lang", type=int, default=None, help="Maximum samples per language (for memory management)"
    )

    # Data/tokenizer
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    # Loss
    parser.add_argument("--lambda_flops", type=float, default=1.0)
    parser.add_argument("--flops_pooling", type=str, choices=["max", "sum"], default="max")

    # Freezing
    parser.add_argument(
        "--freeze_embeddings_and_mlm", action="store_true", help="Freeze all params except embeddings and MLM head"
    )

    # Logging
    parser.add_argument("--wandb_project", type=str, default="mlm-flops")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)

    return parser.parse_args()


def _set_module_trainable(module, requires_grad: bool) -> None:
    for p in module.parameters(recurse=True):
        p.requires_grad = requires_grad


def freeze_all_but_embeddings_and_lm_head(model) -> None:
    # Freeze everything
    _set_module_trainable(model, False)

    # Unfreeze input embeddings
    input_emb = model.get_input_embeddings()
    if input_emb is not None and hasattr(input_emb, "weight"):
        input_emb.weight.requires_grad = True

    # Unfreeze MLM head (try common heads)
    head = None
    if hasattr(model, "cls"):
        head = model.cls
    elif hasattr(model, "lm_head"):
        head = model.lm_head

    if head is not None:
        _set_module_trainable(head, True)
    else:
        # Fallback to decoder returned by get_output_embeddings (at least unfreeze decoder probs and bias)
        out_emb = model.get_output_embeddings()
        if out_emb is not None:
            _set_module_trainable(out_emb, True)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # WandB: rely on HF Trainer integration via report_to and run_name
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Choose data source: multilingual datasets, HF dataset, or local text file
    if args.multilingual_datasets is not None:
        args.multilingual_datasets = [d for d in args.multilingual_datasets if d != " "]
        print(args.multilingual_datasets)
        # Use MultilingualDataset for training on multiple languages
        dataset = MultilingualDataset(
            dataset_paths=args.multilingual_datasets,
            text_field=args.hf_text_field,
            tokenizer=tokenizer,
            max_length=args.max_length,
            balance_languages=args.balance_languages,
            max_samples_per_lang=args.max_samples_per_lang,
        )
    elif args.hf_dataset is not None:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset, args.hf_subset, split=args.hf_split)
        dataset = HFDatasetText(
            hf_dataset=ds, text_field=args.hf_text_field, tokenizer=tokenizer, max_length=args.max_length
        )
    elif args.train_file is not None:
        # Use the new TextFileDataset for text files
        dataset = TextFileDataset(text_file=args.train_file, tokenizer=tokenizer, max_length=args.max_length)
    else:
        raise ValueError("Either --multilingual_datasets, --hf_dataset, or --train_file must be provided")

    collator = SimpleMlmc(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    model = AutoModelForMaskedLM.from_pretrained(args.model)

    if args.freeze_embeddings_and_mlm:
        freeze_all_but_embeddings_and_lm_head(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.wandb_run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        bf16=args.bf16,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=["wandb"],
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    trainer = MlmTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
        lambda_flops=args.lambda_flops,
        flops_pooling=args.flops_pooling,
    )
    # doing this to log before wandb callback
    trainer.callback_handler.callbacks = [MergeAndRoundLogsCallback(trainer)] + trainer.callback_handler.callbacks

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
