"""Neural network definitions shared by the DP-WGAN training and evaluation tools."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def group_count(channels: int) -> int:
    """Return a GroupNorm group count that divides ``channels``."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionalGenerator(nn.Module):
    """32x32 conditional generator used by DP-CWGAN-CP.

    The generator never reads private examples directly. Labels are embedded and
    concatenated with the latent vector before convolutional upsampling.
    """

    def __init__(
        self,
        latent_size: int = 128,
        label_embedding_size: int = 32,
        features: int = 64,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.label_embedding_size = label_embedding_size
        self.features = features
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, label_embedding_size)
        self.project = nn.Sequential(
            nn.Linear(latent_size + label_embedding_size, features * 4 * 4 * 4),
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

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        embedded = self.label_embedding(labels)
        hidden = self.project(torch.cat((noise, embedded), dim=1))
        hidden = hidden.view(noise.size(0), self.features * 4, 4, 4)
        return self.upsample(hidden)


class ProjectionCritic(nn.Module):
    """Auxiliary-classifier projection critic for 32x32 MNIST.

    The critic is the only model that reads private image/label records and is
    therefore the model wrapped by Opacus. The final 4x4 convolution sees the
    full image; the auxiliary head supplies a direct class-learning signal.
    """

    def __init__(self, features: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        feature_dim = features * 8
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features * 4, feature_dim, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.unconditional = nn.Linear(feature_dim, 1)
        self.label_embedding = nn.Embedding(num_classes, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        compute_class_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.encoder(images).flatten(1)
        unconditional_score = self.unconditional(features).squeeze(1)
        conditional_score = (
            self.label_embedding(labels.long()) * features
        ).sum(dim=1) / math.sqrt(self.feature_dim)
        # Skipping this head when its output is unused is important for Opacus:
        # every hooked forward through a trainable layer must participate in the
        # matching backward pass so that per-sample gradients are initialized.
        class_logits = self.classifier(features) if compute_class_logits else None
        return unconditional_score + conditional_score, class_logits


class LeNetEvaluator(nn.Module):
    """MNIST classifier exposing a compact feature vector for LeNet-FID."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 84),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(84, num_classes)

    def forward(
        self, images: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.feature_head(self.features(images))
        logits = self.classifier(embeddings)
        if return_features:
            return logits, embeddings
        return logits


class ConvNetEvaluator(nn.Module):
    """Independent evaluator with a different topology from LeNet."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self, images: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.feature_head(self.features(images))
        logits = self.classifier(embeddings)
        if return_features:
            return logits, embeddings
        return logits


def initialize_gan_weights(module: nn.Module) -> None:
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
