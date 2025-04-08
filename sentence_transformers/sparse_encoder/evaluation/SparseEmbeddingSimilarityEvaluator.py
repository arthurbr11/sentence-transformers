from __future__ import annotations

import csv
import logging
import os
from typing import Literal

from scipy.stats import pearsonr, spearmanr

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from sentence_transformers.util import (
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)

logger = logging.getLogger(__name__)


class SparseEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    """
    Evaluate a sparse model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    Args:
        sentences1 (List[str]): List with the first sentence in a pair.
        sentences2 (List[str]): List with the second sentence in a pair.
        scores (List[float]): Similarity score between sentences1[i] and sentences2[i].
        batch_size (int, optional): The batch size for processing the sentences. Defaults to 16.
        main_similarity (Optional[Union[str, SimilarityFunction]], optional): The main similarity function to use.
            Can be a string (e.g. "cosine", "dot") or a SimilarityFunction object. Defaults to None.
        similarity_fn_names (List[str], optional): List of similarity function names to use. If None, the
            ``similarity_fn_name`` attribute of the model is used. Defaults to None.
        name (str, optional): The name of the evaluator. Defaults to "".
        show_progress_bar (bool, optional): Whether to show a progress bar during evaluation. Defaults to False.
        write_csv (bool, optional): Whether to write the evaluation results to a CSV file. Defaults to True.
        precision (Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]], optional): The precision
            to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". Defaults to None.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the
            model's current truncation dimension. Defaults to None.

    Example:
        ::

            from datasets import load_dataset
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
            eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

            # Initialize the evaluator
            dev_evaluator = EmbeddingSimilarityEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                scores=eval_dataset["score"],
                name="sts_dev",
            )
            results = dev_evaluator(model)
            '''
            EmbeddingSimilarityEvaluator: Evaluating the model on the sts-dev dataset:
            Cosine-Similarity :  Pearson: 0.8806 Spearman: 0.8810
            '''
            print(dev_evaluator.primary_metric)
            # => "sts_dev_pearson_cosine"
            print(results[dev_evaluator.primary_metric])
            # => 0.881019449484294
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        batch_size: int = 16,
        main_similarity: str | SimilarityFunction | None = None,
        similarity_fn_names: (list[Literal["cosine", "euclidean", "manhattan", "dot"]] | None) = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: (Literal["float32", "int8", "uint8", "binary", "ubinary"] | None) = None,
    ):
        super().__init__(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            batch_size=batch_size,
            main_similarity=main_similarity,
            similarity_fn_names=similarity_fn_names,
            name=name,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            precision=precision,
        )

    def __call__(
        self,
        model: SparseEncoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if not isinstance(model, SparseEncoder):
            raise ValueError("This evaluator is designed for SparseEncoder models only")

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"SparseEmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        # Get sparse embeddings
        embeddings1 = model.encode(
            self.sentences1,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor_sparse=True,
        )
        embeddings2 = model.encode(
            self.sentences2,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor_sparse=True,
        )

        labels = self.scores

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda x, y: pairwise_cos_sim(x, y),
            "manhattan": lambda x, y: pairwise_manhattan_sim(x, y),
            "euclidean": lambda x, y: pairwise_euclidean_sim(x, y),
            "dot": lambda x, y: pairwise_dot_score(x, y),
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                scores = similarity_functions[fn_name](embeddings1, embeddings2)
                eval_pearson, _ = pearsonr(labels, scores.cpu().numpy())
                eval_spearman, _ = spearmanr(labels, scores.cpu().numpy())
                metrics[f"pearson_{fn_name}"] = eval_pearson
                metrics[f"spearman_{fn_name}"] = eval_spearman
                logger.info(
                    f"{fn_name.capitalize()}-Similarity :\tPearson: {eval_pearson:.4f}\tSpearman: {eval_spearman:.4f}"
                )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(
                csv_path,
                newline="",
                mode="a" if output_file_exists else "w",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                    ]
                    + [
                        metrics[f"{metric}_{fn_name}"]
                        for fn_name in self.similarity_fn_names
                        for metric in ["pearson", "spearman"]
                    ]
                )

        if len(self.similarity_fn_names) > 1:
            metrics["pearson_max"] = max(metrics[f"pearson_{fn_name}"] for fn_name in self.similarity_fn_names)
            metrics["spearman_max"] = max(metrics[f"spearman_{fn_name}"] for fn_name in self.similarity_fn_names)

        if self.main_similarity:
            self.primary_metric = {
                SimilarityFunction.COSINE: "spearman_cosine",
                SimilarityFunction.EUCLIDEAN: "spearman_euclidean",
                SimilarityFunction.MANHATTAN: "spearman_manhattan",
                SimilarityFunction.DOT_PRODUCT: "spearman_dot",
            }.get(self.main_similarity)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "spearman_max"
            else:
                self.primary_metric = f"spearman_{self.similarity_fn_names[0]}"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
