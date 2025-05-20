from __future__ import annotations

import logging

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments

logger = logging.getLogger(__name__)


class LossComponentLoggingCallback(TrainerCallback):
    """
    A TrainerCallback that logs custom loss components accumulated in the trainer.
    It hooks into the `on_log` event to add averaged custom components to the
    standard logs and then resets the accumulators on the trainer.
    """

    def __init__(self, trainer_ref=None):
        self.trainer_ref = trainer_ref

    def on_log(
        self,
        args: SparseEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs=None,
        **kwargs,
    ):
        """
        Called when the Trainer logs metrics. Adds accumulated custom loss
        components to the logs and resets the accumulators.
        """
        prefix = "eval_" if control.should_evaluate else ""
        averaged_loss_components_over_steps = {
            f"{prefix}custom_{k_sum}": s_sum / self.trainer_ref.loss_components_count
            for k_sum, s_sum in self.trainer_ref.loss_components_sum.items()
        }
        if averaged_loss_components_over_steps:
            logs.update(averaged_loss_components_over_steps)

        # Reset accumulators on the trainer instance
        self.trainer_ref.loss_components_sum = {}
        self.trainer_ref.loss_components_count = 0
