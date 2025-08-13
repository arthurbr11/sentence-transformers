from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


def fa_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )


def _splade_pool_logits(logits: torch.Tensor, attention_mask: torch.Tensor, mode: str = "max") -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(logits.dtype)
    x = torch.relu(logits * mask)
    x = torch.log1p(x)
    if mode == "max":
        pooled = x.max(dim=1).values
    else:
        pooled = x.sum(dim=1)
    return pooled


def flops_from_logits(logits: torch.Tensor, attention_mask: torch.Tensor, mode: str = "max") -> torch.Tensor:
    pooled = _splade_pool_logits(logits, attention_mask, mode)
    return torch.sum(torch.mean(pooled, dim=0) ** 2)


class MlmTrainer(Trainer):
    def __init__(self, *args, lambda_flops: float = 1.0, flops_pooling: str = "max", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_flops = float(lambda_flops)
        self.flops_pooling = flops_pooling
        self._last_ce: float | None = None
        self._last_flops: float | None = None

    def compute_loss(self, model, inputs: dict, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        ce_loss = fa_cross_entropy(logits=logits, labels=labels)
        flops_loss = flops_from_logits(logits=logits, attention_mask=attention_mask, mode=self.flops_pooling)
        loss = ce_loss + self.lambda_flops * flops_loss

        # Persist last metrics; will be merged into logs by callback
        self._last_ce = float(ce_loss.detach().cpu().item())
        self._last_flops = float((self.lambda_flops * flops_loss.detach()).cpu().item())

        return (loss, outputs) if return_outputs else loss


class MergeAndRoundLogsCallback(TrainerCallback):
    def __init__(self, trainer: MlmTrainer, ndigits: int = 5) -> None:
        super().__init__()
        self.trainer = trainer
        self.ndigits = ndigits
        self._last_logged_step: int | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs["loss/ce"] = round(self.trainer._last_ce, self.ndigits)
        logs["loss/flops"] = round(self.trainer._last_flops, self.ndigits)
