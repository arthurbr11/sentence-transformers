from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


class HFDatasetText(Dataset):
    def __init__(self, hf_dataset, text_field: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 128) -> None:
        self.hf_dataset = hf_dataset
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.hf_dataset[idx][self.text_field]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


class TextFileDataset(Dataset):
    """Dataset for reading text files where each line is a training example."""

    def __init__(self, text_file: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read all lines from the text file
        with open(text_file, encoding="utf-8") as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.texts)} texts from {text_file}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


class MultilingualDataset(Dataset):
    """Dataset that combines multiple language datasets for multilingual training."""

    def __init__(
        self,
        dataset_paths: list[str],
        text_field: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        balance_languages: bool = False,
        max_samples_per_lang: int = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field

        # Load all datasets
        datasets = []
        language_names = []

        for dataset_path in dataset_paths:
            try:
                # Try to load as HF dataset directory
                if os.path.isdir(dataset_path):
                    hf_dataset = load_from_disk(dataset_path)
                    lang_name = os.path.basename(dataset_path)
                else:
                    # Assume it's a dataset name/path
                    from datasets import load_dataset

                    lang_name = dataset_path.split("/")[-1]

                    hf_dataset = load_dataset("/".join(dataset_path.split("/")[:-1]), f"collection-{lang_name}")[
                        "collection"
                    ]

                # Limit samples per language if specified
                if max_samples_per_lang and len(hf_dataset) > max_samples_per_lang:
                    hf_dataset = hf_dataset.shuffle(seed=42).select(range(max_samples_per_lang))

                datasets.append(hf_dataset)
                language_names.append(lang_name)
                print(f"Loaded {len(hf_dataset)} samples from {lang_name}")

            except Exception as e:
                print(f"Error loading dataset {dataset_path}: {e}")
                continue

        if not datasets:
            raise ValueError("No valid datasets were loaded!")

        # Balance datasets if requested
        if balance_languages:
            min_size = min(len(ds) for ds in datasets)
            datasets = [ds.shuffle(seed=42).select(range(min_size)) for ds in datasets]
            print(f"Balanced all languages to {min_size} samples each")

        # Concatenate all datasets
        self.combined_dataset = concatenate_datasets(datasets)

        # Shuffle the combined dataset
        self.combined_dataset = self.combined_dataset.shuffle(seed=42)

        print(f"Combined dataset size: {len(self.combined_dataset)}")
        print(f"Languages included: {language_names}")

    def __len__(self) -> int:
        return len(self.combined_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.combined_dataset[idx][self.text_field]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


@dataclass
class MlmcConfig:
    mlm_probability: float = 0.3


class SimpleMlmc(DataCollatorForLanguageModeling):
    """
    Thin wrapper to expose config while reusing HF's masking implementation.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = 0.3) -> None:
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return super().__call__(examples)
