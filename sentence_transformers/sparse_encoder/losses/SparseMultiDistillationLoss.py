from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.sparse_encoder.losses import SparseDistillKLDivLoss, SparseMarginMSELoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMultiDistillationLoss(nn.Module):
    def __init__(
        self, model: SparseEncoder, similarity_fct=util.pairwise_dot_score, kl_div_temperature=1.0, gamma=0.05
    ) -> None:
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.margin_mse_loss = SparseMarginMSELoss(model=model, similarity_fct=similarity_fct)
        self.kl_div_loss = SparseDistillKLDivLoss(
            model=model, similarity_fct=similarity_fct, temperature=kl_div_temperature
        )
        self.gamma = gamma  # Margin for the margin MSE loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return {
            "margin_mse_loss": self.margin_mse_loss.compute_loss_from_embeddings(embeddings, labels) * self.gamma,
            "kl_div_loss": self.kl_div_loss.compute_loss_from_embeddings(embeddings, labels),
        }
