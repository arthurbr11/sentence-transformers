from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.training_args import SentenceTransformerTrainingArguments


@dataclass
class SparseEncoderTrainingArguments(SentenceTransformerTrainingArguments):
    """
    SparseEncoderTrainingArguments extends :class:`~SentenceTransformerTrainingArguments` which itself extend
    :class:`~transformers.TrainingArguments` with additional arguments specific to Sentence Transformers.
    See :class:`~transformers.TrainingArguments` for the complete list of available arguments.

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        prompts (`Union[Dict[str, Dict[str, str]], Dict[str, str], str]`, *optional*):
            The prompts to use for each column in the training, evaluation and test datasets. Four formats are accepted:

            1. `str`: A single prompt to use for all columns in the datasets, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            2. `Dict[str, str]`: A dictionary mapping column names to prompts, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            3. `Dict[str, str]`: A dictionary mapping dataset names to prompts. This should only be used if your training/evaluation/test
               datasets are a :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.
            4. `Dict[str, Dict[str, str]]`: A dictionary mapping dataset names to dictionaries mapping column names to
               prompts. This should only be used if your training/evaluation/test datasets are a
               :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.

        batch_sampler (Union[:class:`~sentence_transformers.training_args.BatchSamplers`, `str`], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`, `str`], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
        log_loss_component (`bool`, *optional*):
            Wether to log the different commponents separately that compose the loss function. It can be great to monitor
            the influence of each component on the final loss.
    """

    log_loss_component: bool = False
