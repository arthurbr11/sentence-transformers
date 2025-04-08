from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
from packaging.version import parse as parse_version
from torch import nn
from transformers import EvalPrediction, PreTrainedTokenizerBase, TrainerCallback
from transformers import __version__ as transformers_version
from transformers.integrations import WandbCallback

from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
from sentence_transformers.sparse_encoder.data_collator import SparseEncoderDataCollator
from sentence_transformers.sparse_encoder.losses import (
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from sentence_transformers.sparse_encoder.training_args import (
    SparseEncoderTrainingArguments,
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, Value

logger = logging.getLogger(__name__)


class SparseEncoderTrainer(SentenceTransformerTrainer):
    """
    SparseEncoderTrainer is a simple but feature-complete training and eval loop for PyTorch
    based on the SentenceTransformerTrainer that based on 🤗 Transformers :class:`~transformers.Trainer`.

    This trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

    - :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if `wandb` is installed
    - :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if `tensorboard` is accessible.
    - :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if `codecarbon` is installed.

        - Note: These carbon emissions will be included in your automatically generated model card.

    See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
    documentation for more information on the integrated callbacks and how to write your own callbacks.

    Args:
        model (:class:`~sentence_transformers.SparseEncoder`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
        args (:class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments`, *optional*):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        train_dataset (Union[:class:`datasets.Dataset`, :class:`datasets.DatasetDict`, :class:`datasets.IterableDataset`, Dict[str, :class:`datasets.Dataset`]], *optional*):
            The dataset to use for training. Must have a format accepted by your loss function, see
            `Training Overview > Dataset Format <../../../docs/sentence_transformer/training_overview.html#dataset-format>`_.
        eval_dataset (Union[:class:`datasets.Dataset`, :class:`datasets.DatasetDict`, :class:`datasets.IterableDataset`, Dict[str, :class:`datasets.Dataset`]], *optional*):
            The dataset to use for evaluation. Must have a format accepted by your loss function, see
            `Training Overview > Dataset Format <../../../docs/sentence_transformer/training_overview.html#dataset-format>`_.
        loss (Optional[Union[:class:`torch.nn.Module`, Dict[str, :class:`torch.nn.Module`],\
            Callable[[:class:`~sentence_transformers.SparseEncoder`], :class:`torch.nn.Module`],\
            Dict[str, Callable[[:class:`~sentence_transformers.SparseEncoder`]]]], *optional*):
            The loss function to use for training. Can either be a loss class instance, a dictionary mapping
            dataset names to loss class instances, a function that returns a loss class instance given a model,
            or a dictionary mapping dataset names to functions that return a loss class instance given a model.
            In practice, the latter two are primarily used for hyper-parameter optimization. Will default to
            :class:`~sentence_transformers.losses.CoSENTLoss` if no ``loss`` is provided.
        evaluator (Union[:class:`~sentence_transformers.evaluation.SentenceEvaluator`,\
            List[:class:`~sentence_transformers.evaluation.SentenceEvaluator`]], *optional*):
            The evaluator instance for useful evaluation metrics during training. You can use an ``evaluator`` with
            or without an ``eval_dataset``, and vice versa. Generally, the metrics that an ``evaluator`` returns
            are more useful than the loss value returned from the ``eval_dataset``. A list of evaluators will be
            wrapped in a :class:`~sentence_transformers.evaluation.SequentialEvaluator` to run them sequentially.
        callbacks (List of [:class:`transformers.TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[:class:`torch.optim.Optimizer`, :class:`torch.optim.lr_scheduler.LambdaLR`]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of :class:`torch.optim.AdamW`
            on your model and a scheduler given by :func:`transformers.get_linear_schedule_with_warmup` controlled by `args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)
    """

    def __init__(
        self,
        model: SparseEncoder | None = None,
        args: SparseEncoderTrainingArguments = None,
        train_dataset: Dataset | DatasetDict | dict[str, Dataset] | None = None,
        eval_dataset: Dataset | DatasetDict | dict[str, Dataset] | None = None,
        loss: (
            nn.Module
            | dict[str, nn.Module]
            | Callable[[SparseEncoder], torch.nn.Module]
            | dict[str, Callable[[SparseEncoder], torch.nn.Module]]
            | None
        ) = None,
        evaluator: SentenceEvaluator | list[SentenceEvaluator] | None = None,
        data_collator: SparseEncoderDataCollator | None = None,
        tokenizer: PreTrainedTokenizerBase | Callable | None = None,
        model_init: Callable[[], SparseEncoder] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None) = None,
    ) -> None:
        if not is_training_available():
            raise RuntimeError(
                "To train a SparseEncoder model, you need to install the `accelerate` and `datasets` modules. "
                "You can do so with the `train` extra:\n"
                'pip install -U "sentence-transformers[train]"'
            )

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `SparseEncoderTrainingArguments` passed, using `output_dir={output_dir}`.")
            args = SparseEncoderTrainingArguments(output_dir=output_dir)
        elif not isinstance(args, SparseEncoderTrainingArguments):
            raise ValueError("Please use `SparseEncoderTrainingArguments` imported from `sentence_transformers`.")

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                logger.warning(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method."
                )
            self.model_init = model_init

        if compute_metrics is not None:
            logger.warning(
                "`compute_metrics` is currently not compatible with the SparseEncoderTrainer. Please use the "
                "`evaluator` argument instead for detailed evaluation metrics, or the `eval_dataset` argument for "
                "the evaluation loss."
            )

        # Get a dictionary of the default training arguments, so we can determine which arguments have been changed
        # for the model card
        default_args_dict = SparseEncoderTrainingArguments(output_dir="unused").to_dict()

        # If the model ID is set via the SparseEncoderTrainingArguments, but not via the SparseEncoderModelCardData,
        # then we can set it here for the model card regardless
        if args.hub_model_id and not model.model_card_data.model_id:
            model.model_card_data.set_model_id(args.hub_model_id)

        if tokenizer is None and isinstance(model.tokenizer, PreTrainedTokenizerBase):
            tokenizer = model.tokenizer

        if data_collator is None:
            data_collator = SparseEncoderDataCollator(
                tokenize_fn=model.tokenize,
            )

        for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
            if isinstance(dataset, IterableDataset) and dataset.column_names is None:
                sample = next(iter(dataset))
                naive_type_mapping = {
                    str: "string",
                    int: "int64",
                    float: "float32",
                    bool: "bool",
                }
                example_features = {
                    key: Value(naive_type_mapping.get(type(value), "null")) for key, value in sample.items()
                }
                raise ValueError(
                    f"The provided `{dataset_name}_dataset` must have Features. Specify them with e.g.:\n"
                    f"{dataset_name}_dataset = {dataset_name}_dataset.cast(Features({example_features}))\n"
                    "or by providing the Features to the IterableDataset initialization method. See the Datasets "
                    "documentation for more information on dataset Features: "
                    "https://huggingface.co/docs/datasets/en/about_dataset_features"
                )

        if isinstance(train_dataset, dict) and not isinstance(train_dataset, DatasetDict):
            train_dataset = DatasetDict(train_dataset)
        if isinstance(eval_dataset, dict) and not isinstance(eval_dataset, DatasetDict):
            eval_dataset = DatasetDict(eval_dataset)

        # Transformers v4.46.0 introduced a ValueError if `eval_dataset` is None while eval_strategy is not "no",
        # but in Sentence Transformers you can also evaluate without an eval_dataset via an evaluator, so we set
        # it to "dummy" in that case to avoid the ValueError
        super_kwargs = {
            "model": None if self.model_init else model,
            "args": args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": (eval_dataset if eval_dataset is not None or evaluator is None else "dummy"),
            "model_init": model_init,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        }
        # Transformers v4.46.0 changed the `tokenizer` argument to a more general `processing_class` argument
        if parse_version(transformers_version) >= parse_version("4.46.0"):
            super_kwargs["processing_class"] = tokenizer
        else:
            super_kwargs["tokenizer"] = tokenizer

        # super.__init__() will still raise a ValueError if `eval_dataset` is None, `evaluator` is None,
        # while eval_strategy is not "no", so let's get ahead of it with a more useful ST-specific error message
        if eval_dataset is None and evaluator is None and args.eval_strategy != "no":
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy}, but you didn't provide an `eval_dataset` or an `evaluator`. "
                "Either provide an `eval_dataset` or an `evaluator` to `SparseEncoderTrainer`, "
                "or set `args.eval_strategy='no'` to skip evaluation."
            )

        # Call the __init__ from Trainer, not from SentenceTransformerTrainer
        super(SentenceTransformerTrainer, self).__init__(**super_kwargs)
        # If the eval_dataset is "dummy", then we set it back to None
        if self.eval_dataset == "dummy":
            self.eval_dataset = None

        # Every Sentence Transformer model can always return a loss, so we set this to True
        # to avoid having to specify it in the data collator or model's forward
        self.can_return_loss = True

        self._prompt_length_mapping = {}

        self.model: SparseEncoder
        self.args: SparseEncoderTrainingArguments
        self.data_collator: SparseEncoderDataCollator
        # Set the W&B project via environment variables if it's not already set
        if any([isinstance(callback, WandbCallback) for callback in self.callback_handler.callbacks]):
            os.environ.setdefault("WANDB_PROJECT", "sentence-transformers")

        if loss is None:
            logger.info(
                "No `loss` passed, using `sparse_encoder.losses.SparseMultipleNegativesRankingLoss` as a default option."
            )
            loss = SparseMultipleNegativesRankingLoss(self.model)

        if isinstance(loss, dict):
            self.loss = {dataset_name: self.prepare_loss(loss_fn, model) for dataset_name, loss_fn in loss.items()}
            for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
                if dataset is None:
                    continue
                if not isinstance(dataset, dict):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then the `{dataset_name}_dataset` must be a `DatasetDict`."
                    )
                if missing := set(dataset.keys()) - set(loss.keys()):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then all keys from the `{dataset_name}_dataset` dictionary must occur in `loss` also. "
                        f"Currently, {sorted(missing)} occur{'s' if len(missing) == 1 else ''} in `{dataset_name}_dataset` but not in `loss`."
                    )
        else:
            self.loss = self.prepare_loss(loss, model)

        # If evaluator is a list, we wrap it in a SequentialEvaluator
        if evaluator is not None and not isinstance(evaluator, SentenceEvaluator):
            evaluator = SequentialEvaluator(evaluator)
        self.evaluator = evaluator

        if self.train_dataset is not None:
            self.train_dataset = self.maybe_add_prompts_or_dataset_name_column(
                train_dataset, args.prompts, dataset_name="train"
            )
        if self.eval_dataset is not None:
            self.eval_dataset = self.maybe_add_prompts_or_dataset_name_column(
                eval_dataset, args.prompts, dataset_name="eval"
            )
        self.add_model_card_callback(default_args_dict)

    def add_model_card_callback(self, default_args_dict: dict[str, Any]) -> None:
        # TODO: Adapth this to the new model card system for sparse encoders
        pass
