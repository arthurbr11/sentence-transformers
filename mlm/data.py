from __future__ import annotations

from dataclasses import dataclass

import torch
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
