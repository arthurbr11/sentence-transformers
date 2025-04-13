from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class CSRSparsity(nn.Module):
    """
    CSR (Compressed Sparse Row) Sparsity module.

    This module implements the Sparse AutoEncoder architecture based on the paper:
    Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation, https://arxiv.org/abs/2503.01776
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        k: int = 8,
        k_aux: int = 512,
        normalize: bool = False,
        dead_threshold: int = 30,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dead_threshold = dead_threshold
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder: nn.Module = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder: TiedTranspose = TiedTranspose(self.encoder)
        self.k = k
        self.k_aux = k_aux
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_threshold
            x.data *= dead_mask  # inplace to save memoryr
            return x

        self.auxk_mask_fn = auxk_mask_fn

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, input_dim])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, hidden_dim])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(x, self.encoder.weight, self.latent_bias)
        return latents_pre_act

    def LN(self, x: torch.Tensor, eps: float = 1e-5):
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def preprocess(self, x: torch.Tensor):
        if not self.normalize:
            return x, dict()
        x, mu, std = self.LN(x)
        return x, dict(mu=mu, std=std)

    def top_k(self, x: torch.Tensor, k=None) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, input_dim])
        :return: autoencoder latents (shape: [batch, hidden_dim])
        """
        if k is None:
            k = self.k
        topk = torch.topk(x, k=k, dim=-1)
        z_topk = torch.zeros_like(x)
        z_topk.scatter_(-1, topk.indices, topk.values)
        latents_k = F.relu(z_topk)
        ## set num nonzero stat ##
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            0,
            topk.indices.reshape(-1),
            (topk.values > 1e-5).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1
        ## end stats ##

        if self.k_aux:
            aux_topk = torch.topk(
                input=self.auxk_mask_fn(x),
                k=self.k_aux,
            )
            z_auxk = torch.zeros_like(x)
            z_auxk.scatter_(-1, aux_topk.indices, aux_topk.values)
            latents_auxk = F.relu(z_auxk)
        return latents_k, latents_auxk

    def decode(self, latents: torch.Tensor, info=None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, hidden_dim])
        :return: reconstructed data (shape: [batch, n_inputs])
        """

        ret = self.decoder(latents) + self.pre_bias

        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features["sentence_embedding"]

        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)

        latents_k, latents_auxk = self.top_k(latents_pre_act, self.k)
        latents_4k, _ = self.top_k(latents_pre_act, 4 * self.k)

        recons_k = self.decode(latents_k, info)
        recons_4k = self.decode(latents_4k, info)

        recons_aux = self.decode(latents_auxk, info)

        # Update the features dictionary
        features.update(
            {
                "sentence_embedding_backbone": x,
                "sentence_embedding_encoded": latents_pre_act,
                "sentence_embedding_encoded_4k": latents_4k,
                "auxiliary_embedding": latents_auxk,
                "decoded_embedding_k": recons_k,
                "decoded_embedding_4k": recons_4k,
                "decoded_embedding_aux": recons_aux,
                "decoded_embedding_k_pre_bias": recons_k + self.pre_bias,
            }
        )
        features["sentence_embedding"] = latents_k
        return features

    def save(self, output_path, safe_serialization: bool = True) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        module = CSRSparsity(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(module, os.path.join(input_path, "model.safetensors"))
        else:
            module.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"), weights_only=True
                )
            )
        return module

    def __repr__(self):
        return f"CSRSparsity({self.get_config_dict()})"

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "k": self.k,
            "k_aux": self.k_aux,
            "normalize": self.normalize,
            "dead_threshold": self.dead_threshold,
        }
