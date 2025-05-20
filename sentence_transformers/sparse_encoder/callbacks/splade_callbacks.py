from __future__ import annotations

import logging
from enum import Enum

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from sentence_transformers.sparse_encoder.losses.SpladeLoss import SpladeLoss
from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    """Types of schedulers for lambda parameters in SpladeLoss"""

    LINEAR = "linear"
    QUADRATIC = "quadratic"


class SpladeLambdaSchedulerCallback(TrainerCallback):
    def __init__(
        self,
        loss: SpladeLoss,
        scheduler_type: str | SchedulerType = SchedulerType.QUADRATIC,
        warmup_ratio: float = 1 / 3,
    ):
        """
        Callback that updates the lambda_query and lambda_corpus parameters of SpladeLoss
        based on a schedule.

        The scheduler gradually increases the lambda values from 0 to their max value
        within the specified warmup ratio of the total training steps.

         Args:
                loss: SpladeLoss instance to be updated
                scheduler_type: Type of scheduler ('linear' or 'quadratic')
                warmup_ratio: Ratio of total steps to reach max lambda values (default: 1/3)
        """
        super().__init__()

        if isinstance(scheduler_type, str):
            try:
                scheduler_type = SchedulerType(scheduler_type.lower())
            except ValueError:
                logger.warning(
                    f"Invalid scheduler_type: {scheduler_type}. Using default: {SchedulerType.QUADRATIC.value}"
                )
                scheduler_type = SchedulerType.QUADRATIC

        self.scheduler_type = scheduler_type

        # Validate warmup_ratio is between 0 and 1
        if not 0 < warmup_ratio <= 1:
            logger.warning(f"warmup_ratio should be between 0 and 1, got {warmup_ratio}. Setting to default 1/3.")
            warmup_ratio = 1 / 3

        # Validate loss is an instance of SpladeLoss
        if not isinstance(loss, SpladeLoss):
            logger.warning(
                f"SpladeLambdaSchedulerCallback is only compatible with SpladeLoss, "
                f"but got {type(loss).__name__}. This callback won't have any effect."
            )
            raise ValueError("loss must be an instance of SpladeLoss")
        self.loss = loss
        self.max_lambda_corpus = self.loss.lambda_corpus
        self.max_lambda_query = self.loss.lambda_query
        self.warmup_ratio = warmup_ratio
        self._current_lambda_query = 0.0 if self.max_lambda_query is not None else None
        self._current_lambda_corpus = 0.0
        self.total_steps = None
        self.warmup_steps = None

    def on_train_begin(
        self,
        args: SparseEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize the scheduler at the beginning of training."""
        # Calculate total steps and warmup steps
        if hasattr(state, "max_steps") and state.max_steps > 0:
            self.total_steps = state.max_steps
        elif hasattr(state, "num_train_epochs") and hasattr(state, "num_update_steps_per_epoch"):
            self.total_steps = state.num_update_steps_per_epoch * state.num_train_epochs
        else:
            logger.warning("Cannot determine total steps from TrainerState. Lambda scheduling may not work properly.")
            return

        self.warmup_steps = int(self.total_steps * self.warmup_ratio)
        if self.warmup_steps <= 0:
            self.warmup_steps = 1  # Ensure at least one step for warmup

        # Set initial lambda values
        self.loss.lambda_query = self._current_lambda_query
        self.loss.lambda_corpus = self._current_lambda_corpus

    def _calculate_lambda_value(self, step: int, max_value: float) -> float:
        """Calculate the lambda value based on the current step and scheduler type."""
        if self.warmup_steps is None or step >= self.warmup_steps or max_value is None:
            return max_value

        ratio = step / max(self.warmup_steps, 1)  # Avoid division by zero

        if self.scheduler_type == SchedulerType.LINEAR:
            return max_value * ratio
        elif self.scheduler_type == SchedulerType.QUADRATIC:
            return max_value * (ratio**2)
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}. Using quadratic.")
            return max_value * (ratio**2)

    def on_step_begin(
        self,
        args: SparseEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Update lambda values at the end of each step."""
        if self.total_steps is None or self.warmup_steps is None:
            return

        # Get current step
        step = state.global_step

        # Calculate new lambda values
        new_lambda_query = self._calculate_lambda_value(step, self.max_lambda_query)
        new_lambda_corpus = self._calculate_lambda_value(step, self.max_lambda_corpus)

        # Update lambda values only if they've changed
        if new_lambda_query != self._current_lambda_query or new_lambda_corpus != self._current_lambda_corpus:
            self.loss.lambda_query = new_lambda_query
            self.loss.lambda_corpus = new_lambda_corpus

            # Store current values
            self._current_lambda_query = new_lambda_query
            self._current_lambda_corpus = new_lambda_corpus

    def on_log(
        self,
        args: SparseEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs=None,
        **kwargs,
    ):
        """Log the current lambda values."""
        logs["lambda_corpus"] = self._current_lambda_corpus
        if self._current_lambda_query is not None:
            logs["lambda_query"] = self._current_lambda_query
