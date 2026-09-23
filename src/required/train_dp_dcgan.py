"""Train a DCGAN discriminator with Opacus differential privacy on MNIST.

This is a reproducible command-line implementation of the mandatory experiment
from Assignment.ipynb. Only the discriminator sees private training examples;
the generator learns through the discriminator, which is post-processing of the
private mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from tqdm import tqdm


@dataclass
class EpochMetrics:
    epoch: int
    discriminator_loss: float
    generator_loss: float
    d_real: float
    d_fake_before_g: float
    d_fake_after_g: float
    epsilon: float
    delta: float
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mandatory Experiment 8: DP-SGD + DCGAN on one MNIST digit"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/required"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--generator-features", type=int, default=32)
    parser.add_argument("--discriminator-features", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--target-digit", type=int, choices=range(10), default=8)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic subset size, useful only for smoke tests",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        help="Run a non-private baseline; the mandatory run must not use this flag",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Generator(nn.Module):
    def __init__(self, latent_size: int, features: int, image_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_size, features * 4, 4, 1, 0, bias=False),
            nn.GroupNorm(group_count(features * 4), features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(group_count(features * 2), features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.GroupNorm(group_count(features), features),
            nn.ReLU(True),
        ]
        if image_size == 64:
            layers.extend(
                [
                    nn.ConvTranspose2d(features, features // 2, 4, 2, 1, bias=False),
                    nn.GroupNorm(group_count(features // 2), features // 2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(features // 2, 1, 4, 2, 1, bias=False),
                ]
            )
        else:
            layers.append(nn.ConvTranspose2d(features, 1, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.main(inputs)


class Discriminator(nn.Module):
    def __init__(self, features: int, image_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(1, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(group_count(features * 2), features * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(group_count(features * 4), features * 4),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        if image_size == 64:
            layers.extend(
                [
                    nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
                    nn.GroupNorm(group_count(features * 8), features * 8),
                    nn.LeakyReLU(0.2, inplace=False),
                    nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
                ]
            )
        else:
            layers.append(nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.main(inputs).view(-1)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.GroupNorm):
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def build_dataset(args: argparse.Namespace):
    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    indices = torch.where(dataset.targets == args.target_digit)[0].tolist()
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in order[: args.max_samples]]
    return Subset(dataset, indices)


def save_metrics(path: Path, metrics: list[EpochMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in metrics)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")

    seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    sample_dir = output_dir / "samples"
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    loader_generator = torch.Generator().manual_seed(args.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        generator=loader_generator,
    )

    device = torch.device(args.device)
    generator = Generator(args.latent_size, args.generator_features, args.image_size).to(device)
    discriminator = Discriminator(args.discriminator_features, args.image_size).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    errors = ModuleValidator.validate(discriminator, strict=False)
    if errors:
        raise RuntimeError(f"Discriminator is not Opacus-compatible: {errors}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999)
    )
    optimizer_g = optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999)
    )

    privacy_engine: PrivacyEngine | None = None
    if not args.disable_dp:
        # secure_mode=False keeps the educational run reproducible. For a real
        # deployment, install torchcsprng and enable secure mode.
        privacy_engine = PrivacyEngine(secure_mode=False)
        discriminator, optimizer_d, data_loader = privacy_engine.make_private(
            module=discriminator,
            optimizer=optimizer_d,
            data_loader=data_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )

    fixed_noise = torch.randn(64, args.latent_size, 1, 1, device=device)
    metrics: list[EpochMetrics] = []
    run_started = time.time()

    print(
        json.dumps(
            {
                "device": str(device),
                "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "dataset_size": len(dataset),
                "target_digit": args.target_digit,
                "differential_privacy": not args.disable_dp,
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        sums = {"d": 0.0, "g": 0.0, "dx": 0.0, "dg1": 0.0, "dg2": 0.0}
        steps = 0
        progress = tqdm(data_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")

        for real_images, _ in progress:
            if real_images.numel() == 0:
                continue
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # Train the discriminator with real and generated examples.
            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            real_loss = criterion(real_logits, torch.ones_like(real_logits))

            noise = torch.randn(batch_size, args.latent_size, 1, 1, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            optimizer_d.step()

            # Train G through a frozen D. Disabling Opacus hooks prevents this
            # public, generated-only pass from entering D's per-sample gradients.
            optimizer_g.zero_grad(set_to_none=True)
            set_requires_grad(discriminator, False)
            if hasattr(discriminator, "disable_hooks"):
                discriminator.disable_hooks()
            generator_logits = discriminator(fake_images)
            generator_loss = criterion(generator_logits, torch.ones_like(generator_logits))
            generator_loss.backward()
            optimizer_g.step()
            if hasattr(discriminator, "enable_hooks"):
                discriminator.enable_hooks()
            set_requires_grad(discriminator, True)

            with torch.no_grad():
                d_real = torch.sigmoid(real_logits).mean().item()
                d_fake_before = torch.sigmoid(fake_logits).mean().item()
                d_fake_after = torch.sigmoid(generator_logits).mean().item()
            sums["d"] += discriminator_loss.item()
            sums["g"] += generator_loss.item()
            sums["dx"] += d_real
            sums["dg1"] += d_fake_before
            sums["dg2"] += d_fake_after
            steps += 1
            progress.set_postfix(loss_d=f"{discriminator_loss.item():.3f}", loss_g=f"{generator_loss.item():.3f}")

        if steps == 0:
            raise RuntimeError("The private data loader produced no batches")
        epsilon = (
            privacy_engine.accountant.get_epsilon(delta=args.delta)
            if privacy_engine is not None
            else float("inf")
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            discriminator_loss=sums["d"] / steps,
            generator_loss=sums["g"] / steps,
            d_real=sums["dx"] / steps,
            d_fake_before_g=sums["dg1"] / steps,
            d_fake_after_g=sums["dg2"] / steps,
            epsilon=epsilon,
            delta=args.delta,
            duration_seconds=time.time() - epoch_started,
        )
        metrics.append(epoch_metrics)
        save_metrics(output_dir / "metrics.csv", metrics)

        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise)
        utils.save_image(
            samples,
            sample_dir / f"generated_epoch_{epoch:03d}.png",
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )
        generator.train()
        torch.save(generator.state_dict(), checkpoint_dir / "generator_last.pt")
        # The discriminator is private training state and is kept only for
        # reproducibility; only the generator should be released publicly.
        torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator_private_last.pt")
        print(json.dumps(asdict(epoch_metrics), ensure_ascii=False))

    summary = {
        "status": "completed",
        "mandatory_task": "DP-SGD + DCGAN",
        "differential_privacy_enabled": not args.disable_dp,
        "privacy_accounting_note": "Educational run with non-cryptographic PRNG (secure_mode=False).",
        "epsilon": metrics[-1].epsilon,
        "delta": args.delta,
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset": "MNIST train split",
        "target_digit": args.target_digit,
        "dataset_size": len(dataset),
        "image_size": args.image_size,
        "seed": args.seed,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "total_duration_seconds": time.time() - run_started,
        "final_metrics": asdict(metrics[-1]),
        "released_model": "checkpoints/generator_last.pt",
        "final_sample": f"samples/generated_epoch_{args.epochs:03d}.png",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
