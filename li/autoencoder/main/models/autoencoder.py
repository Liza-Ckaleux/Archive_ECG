from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn


def _build_encoder(
    input_dim: int,
    layer_dims: Sequence[int],
    dropout: float,
    use_batchnorm: bool = True,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for idx, dim in enumerate(layer_dims):
        layers.append(nn.Linear(prev_dim, dim))
        is_last = idx == len(layer_dims) - 1
        if not is_last:
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        prev_dim = dim
    return nn.Sequential(*layers)


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 128
    encoder_dims: Iterable[int] = (1024, 512, 256)
    decoder_dims: Iterable[int] = (256, 512, 1024)
    dropout: float = 0.1


class ECGAutoencoder(nn.Module):
    """Fully-connected autoencoder for one ECG lead."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()

        encoder_dims = list(config.encoder_dims) + [config.latent_dim]
        decoder_dims = [config.latent_dim] + list(config.decoder_dims) + [config.input_dim]

        self.encoder = _build_encoder(
            input_dim=config.input_dim,
            layer_dims=encoder_dims,
            dropout=config.dropout,
        )

        decoder_layers: List[nn.Module] = []
        prev_dim = decoder_dims[0]
        for dim in decoder_dims[1:]:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if dim != config.input_dim:
                decoder_layers.append(nn.ReLU())
            prev_dim = dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> dict:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return {"reconstruction": reconstruction, "latent": latent}
