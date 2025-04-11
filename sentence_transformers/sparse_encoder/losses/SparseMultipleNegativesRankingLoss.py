from __future__ import annotations

from sentence_transformers import util
from sentence_transformers.losses.MultipleNegativesRankingLoss import (
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    """
    Multiple Negatives Ranking Loss for sparse embeddings.
    This loss function is adapted to work with sparse embeddings produced by the SparseEncoder.
    Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the following:

    1. Given an anchor (e.g. a question), assign the highest similarity to the corresponding positive (i.e. answer)
        out of every single positive and negative (e.g. all answers) in the batch.

    If you provide the optional negatives, they will all be used as extra options from which the model must pick the
    correct positive. Within reason, the harder this "picking" is, the stronger the model will become. Because of
    this, a higher batch size results in more in-batch negatives, which then increases performance (to a point).

    This loss function works great to train embeddings for retrieval setups where you have positive pairs
    (e.g. (query, answer)) as it will sample in each batch ``n-1`` negative docs randomly.

    This loss is also known as InfoNCE loss, SimCSE loss, Cross-Entropy Loss with in-batch negatives, or simply
    in-batch negatives loss.

    Args:
        model: SentenceTransformer model
        scale: Output of similarity function is multiplied by scale
            value
        similarity_fct: similarity function between sentence
            embeddings. By default, cos_sim. Can also be set to dot
            product (and then set scale to 1)

    References:
        - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
        - `Training Examples > Natural Language Inference <../../../examples/sentence_transformer/training/nli/README.html>`_
        - `Training Examples > Paraphrase Data <../../../examples/sentence_transformer/training/paraphrases/README.html>`_
        - `Training Examples > Quora Duplicate Questions <../../../examples/sentence_transformer/training/quora_duplicate_questions/README.html>`_
        - `Training Examples > MS MARCO <../../../examples/sentence_transformer/training/ms_marco/README.html>`_
        - `Unsupervised Learning > SimCSE <../../../examples/sentence_transformer/unsupervised_learning/SimCSE/README.html>`_
        - `Unsupervised Learning > GenQ <../../../examples/sentence_transformer/unsupervised_learning/query_generation/README.html>`_

    Requirements:
        1. (anchor, positive) pairs or (anchor, positive, negative) triplets

    Inputs:
        +-------------------------------------------------+--------+
        | Texts                                           | Labels |
        +=================================================+========+
        | (anchor, positive) pairs                        | none   |
        +-------------------------------------------------+--------+
        | (anchor, positive, negative) triplets           | none   |
        +-------------------------------------------------+--------+
        | (anchor, positive, negative_1, ..., negative_n) | none   |
        +-------------------------------------------------+--------+

    Recommendations:
        - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
            ensure that no in-batch negatives are duplicates of the anchor or positive samples.

    Relations:
        - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
            much higher batch sizes (and thus better performance) without extra memory usage. However, it is slightly
            slower.
        - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
        - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
            sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

    Example:
        ::

            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
            from datasets import Dataset

            model = SentenceTransformer("microsoft/mpnet-base")
            train_dataset = Dataset.from_dict({
                "anchor": ["It's nice weather outside today.", "He drove to work."],
                "positive": ["It's so sunny.", "He took the car to the office."],
            })
            loss = losses.MultipleNegativesRankingLoss(model)

            trainer = SentenceTransformerTrainer(
                model=model,
                train_dataset=train_dataset,
                loss=loss,
            )
            trainer.train()
    """

    def __init__(self, model: SparseEncoder, scale: float = 20.0, similarity_fct=util.cos_sim) -> None:
        super().__init__(model, scale, similarity_fct)
