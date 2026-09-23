"""Models used by the additional DP-VAE and single-digit DP-WGAN-CP experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionalConvVAE(nn.Module):
    """A 32x32 conditional convolutional VAE for MNIST."""

    def __init__(
        self,
        latent_size: int = 64,
        label_embedding_size: int = 32,
        features: int = 32,
        num_classes: int = 10,
        prior_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.label_embedding_size = label_embedding_size
        self.features = features
        self.num_classes = num_classes
        self.prior_scale = prior_scale
        if latent_size < num_classes:
            raise ValueError("latent_size must be at least num_classes")

        self.encoder_label = nn.Embedding(num_classes, 32 * 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.GroupNorm(group_count(features * 2), features * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.GroupNorm(group_count(features * 4), features * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
        )
        encoded_size = features * 4 * 4 * 4
        self.mu = nn.Linear(encoded_size, latent_size)
        self.logvar = nn.Linear(encoded_size, latent_size)

        self.decoder_label = nn.Embedding(num_classes, label_embedding_size)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_size + label_embedding_size, encoded_size),
            nn.ReLU(inplace=False),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1),
            nn.GroupNorm(group_count(features * 2), features * 2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1),
            nn.GroupNorm(group_count(features), features),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(features, 1, 4, 2, 1),
        )

    def encode(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        label_map = self.encoder_label(labels.long()).view(-1, 1, 32, 32)
        hidden = self.encoder(torch.cat((images, label_map), dim=1))
        return self.mu(hidden), self.logvar(hidden).clamp(-12.0, 12.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def prior_mean(self, labels: torch.Tensor) -> torch.Tensor:
        means = torch.zeros(labels.size(0), self.latent_size, device=labels.device)
        return means.scatter(1, labels.long().unsqueeze(1), self.prior_scale)

    def decode_logits(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedded = self.decoder_label(labels.long())
        hidden = self.decoder_input(torch.cat((latent, embedded), dim=1))
        hidden = hidden.view(latent.size(0), self.features * 4, 4, 4)
        return self.decoder(hidden)

    def decode(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(latent, labels))

    def generate(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        conditioned_latent = latent + self.prior_mean(labels)
        return self.decode(conditioned_latent, labels).mul(2.0).sub(1.0)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images, labels)
        latent = self.reparameterize(mu, logvar)
        return self.decode_logits(latent, labels), mu, logvar, self.prior_mean(labels)


class SingleDigitGenerator(nn.Module):
    """Unconditional 32x32 generator for a single MNIST digit."""

    def __init__(self, latent_size: int = 128, features: int = 64) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.features = features
        self.project = nn.Sequential(
            nn.Linear(latent_size, features * 4 * 4 * 4),
            nn.ReLU(inplace=False),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(group_count(features * 2), features * 2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.GroupNorm(group_count(features), features),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(features, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        hidden = self.project(noise).view(noise.size(0), self.features * 4, 4, 4)
        return self.upsample(hidden)


class SingleDigitCritic(nn.Module):
    """Normalization-free critic compatible with Opacus."""

    def __init__(self, features: int = 32) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features * 4, 1, 4, 1, 0),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.main(images).flatten()


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.GroupNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
