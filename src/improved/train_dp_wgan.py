"""Train a conditional differentially private WGAN-CP on all MNIST digits.

Only the critic consumes private image/label records. Opacus clips its
per-example gradients, adds Gaussian noise and accounts for privacy. The
generator is trained through the private critic and can be released by the
post-processing property of differential privacy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from dp_wgan_models import (
    ConditionalGenerator,
    ProjectionCritic,
    initialize_gan_weights,
)


@dataclass
class EpochMetrics:
    epoch: int
    critic_loss: float
    critic_classification_loss: float
    critic_real_accuracy: float
    generator_loss: float
    generator_classification_loss: float
    generator_internal_accuracy: float
    wasserstein_estimate: float
    real_score: float
    fake_score: float
    epsilon: float
    delta: float
    noise_multiplier: float
    clip_boundary_fraction: float
    critic_updates: int
    generator_updates: int
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional DP-WGAN-CP on the complete MNIST 0-9 dataset"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/improved"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--label-embedding-size", type=int, default=32)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--critic-features", type=int, default=32)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    parser.add_argument(
        "--wasserstein-weight",
        type=float,
        default=0.01,
        help="Scale of the Wasserstein term relative to auxiliary class loss",
    )
    parser.add_argument("--critic-class-weight", type=float, default=0.2)
    parser.add_argument("--generator-class-weight", type=float, default=1.0)
    parser.add_argument(
        "--critic-steps",
        type=int,
        default=5,
        help="Update G once after this many private critic batches",
    )
    parser.add_argument(
        "--weight-clip",
        type=float,
        default=0.01,
        help="WGAN-CP parameter clipping bound; unrelated to DP gradient clipping",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help="Explicit sigma; when provided it overrides --target-epsilon calibration",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Deterministic subset for smoke checks only; omit for formal training",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        help="Non-private ablation only; never use for the reported DP result",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "epochs",
        "batch_size",
        "latent_size",
        "label_embedding_size",
        "generator_features",
        "critic_features",
        "generator_lr",
        "critic_lr",
        "wasserstein_weight",
        "critic_class_weight",
        "generator_class_weight",
        "critic_steps",
        "weight_clip",
        "max_grad_norm",
        "delta",
        "save_every",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.noise_multiplier is not None and args.noise_multiplier <= 0:
        raise ValueError("--noise-multiplier must be positive")
    if args.target_epsilon <= 0:
        raise ValueError("--target-epsilon must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.max_samples is not None and args.max_samples < 10:
        raise ValueError("--max-samples must be at least 10")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataset(args: argparse.Namespace):
    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    if args.max_samples is None:
        return dataset
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.max_samples]
    return Subset(dataset, indices.tolist())


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def update_ema(ema_model: nn.Module, source_model: nn.Module, decay: float) -> None:
    for ema_parameter, source_parameter in zip(
        ema_model.parameters(), source_model.parameters()
    ):
        ema_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for ema_buffer, source_buffer in zip(ema_model.buffers(), source_model.buffers()):
        ema_buffer.copy_(source_buffer)


def unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_module", module)


@torch.no_grad()
def clip_wasserstein_weights(critic: nn.Module, bound: float) -> float:
    """Clip only weights that affect the Wasserstein score.

    Biases do not affect Lipschitz continuity with respect to the input, and the
    auxiliary classifier is not part of the Wasserstein scalar. Excluding them
    prevents the class head and score offset from being crushed by WGAN-CP.
    Returns the fraction of clipped score parameters at the boundary.
    """
    boundary = 0
    total = 0
    for name, parameter in unwrap_module(critic).named_parameters():
        if name.startswith("classifier.") or not name.endswith("weight"):
            continue
        parameter.clamp_(-bound, bound)
        boundary += parameter.abs().ge(bound * 0.999).sum().item()
        total += parameter.numel()
    return boundary / total if total else 0.0


def save_csv(path: Path, rows: list[EpochMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def checkpoint_payload(
    generator: ConditionalGenerator,
    ema_generator: ConditionalGenerator,
    critic: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    epsilon: float,
    noise_multiplier: float,
    total_duration_seconds: float,
) -> dict[str, Any]:
    model_config = {
        "latent_size": args.latent_size,
        "label_embedding_size": args.label_embedding_size,
        "generator_features": args.generator_features,
        "critic_features": args.critic_features,
        "num_classes": 10,
        "image_size": 32,
    }
    return {
        "algorithm": "DP-ACWGAN-CP",
        "epoch": epoch,
        "generator": generator.state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "critic_private": unwrap_module(critic).state_dict(),
        "model_config": model_config,
        "training_config": vars(args),
        "privacy": {
            "enabled": not args.disable_dp,
            "epsilon": epsilon,
            "delta": args.delta,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": args.max_grad_norm,
            "accountant": "RDP",
        },
        "total_duration_seconds": total_duration_seconds,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    seed_everything(args.seed)
    device = torch.device(args.device)
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
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )

    generator = ConditionalGenerator(
        latent_size=args.latent_size,
        label_embedding_size=args.label_embedding_size,
        features=args.generator_features,
    ).to(device)
    critic = ProjectionCritic(features=args.critic_features).to(device)
    generator.apply(initialize_gan_weights)
    critic.apply(initialize_gan_weights)
    ema_generator = copy.deepcopy(generator).eval()
    set_requires_grad(ema_generator, False)

    validation_errors = ModuleValidator.validate(critic, strict=False)
    if validation_errors:
        raise RuntimeError(f"Critic is not Opacus-compatible: {validation_errors}")

    optimizer_g = optim.Adam(
        generator.parameters(), lr=args.generator_lr, betas=(0.0, 0.9)
    )
    optimizer_c = optim.RMSprop(critic.parameters(), lr=args.critic_lr)

    privacy_engine: PrivacyEngine | None = None
    if not args.disable_dp:
        # secure_mode=False is reproducible for coursework. Production usage
        # requires torchcsprng and secure_mode=True.
        privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        if args.noise_multiplier is None:
            critic, optimizer_c, data_loader = privacy_engine.make_private_with_epsilon(
                module=critic,
                optimizer=optimizer_c,
                data_loader=data_loader,
                target_epsilon=args.target_epsilon,
                target_delta=args.delta,
                epochs=args.epochs,
                max_grad_norm=args.max_grad_norm,
            )
        else:
            critic, optimizer_c, data_loader = privacy_engine.make_private(
                module=critic,
                optimizer=optimizer_c,
                data_loader=data_loader,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
            )

    actual_noise_multiplier = float(
        getattr(optimizer_c, "noise_multiplier", 0.0)
    )
    class_criterion = nn.CrossEntropyLoss()
    fixed_labels = torch.arange(10, device=device).repeat_interleave(10)
    fixed_noise = torch.randn(100, args.latent_size, device=device)
    epoch_rows: list[EpochMetrics] = []
    run_started = time.time()
    global_critic_updates = 0

    print(
        json.dumps(
            {
                "algorithm": "DP-ACWGAN-CP",
                "device": str(device),
                "cuda_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "dataset": "MNIST train split, digits 0-9",
                "dataset_size": len(dataset),
                "differential_privacy": not args.disable_dp,
                "noise_multiplier": actual_noise_multiplier,
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        sums = {
            "critic": 0.0,
            "critic_class": 0.0,
            "critic_accuracy": 0.0,
            "generator": 0.0,
            "generator_class": 0.0,
            "generator_accuracy": 0.0,
            "w": 0.0,
            "real": 0.0,
            "fake": 0.0,
            "clip_boundary": 0.0,
        }
        critic_updates = 0
        generator_updates = 0
        progress = tqdm(data_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")

        for real_images, real_labels in progress:
            if real_images.numel() == 0:
                continue
            real_images = real_images.to(device, non_blocking=True)
            real_labels = real_labels.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            set_requires_grad(critic, True)
            optimizer_c.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.latent_size, device=device)
            fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            with torch.no_grad():
                fake_images = generator(noise, fake_labels)
            real_scores, real_class_logits = critic(real_images, real_labels)
            fake_scores, _ = critic(
                fake_images, fake_labels, compute_class_logits=False
            )
            critic_classification_loss = class_criterion(
                real_class_logits, real_labels
            )
            critic_loss = (
                args.wasserstein_weight
                * (fake_scores.mean() - real_scores.mean())
                + args.critic_class_weight * critic_classification_loss
            )
            critic_loss.backward()
            optimizer_c.step()

            # WGAN-CP weight clipping enforces the critic's Lipschitz constraint.
            # This is separate from Opacus per-example gradient clipping.
            clip_boundary_fraction = clip_wasserstein_weights(
                critic, args.weight_clip
            )

            global_critic_updates += 1
            critic_updates += 1
            wasserstein_estimate = real_scores.mean() - fake_scores.mean()
            sums["critic"] += critic_loss.item()
            sums["critic_class"] += critic_classification_loss.item()
            sums["critic_accuracy"] += (
                real_class_logits.argmax(dim=1).eq(real_labels).float().mean().item()
            )
            sums["w"] += wasserstein_estimate.item()
            sums["real"] += real_scores.mean().item()
            sums["fake"] += fake_scores.mean().item()
            sums["clip_boundary"] += clip_boundary_fraction

            if global_critic_updates % args.critic_steps == 0:
                optimizer_g.zero_grad(set_to_none=True)
                set_requires_grad(critic, False)
                if hasattr(critic, "disable_hooks"):
                    critic.disable_hooks()
                noise_g = torch.randn(batch_size, args.latent_size, device=device)
                labels_g = torch.randint(0, 10, (batch_size,), device=device)
                generated = generator(noise_g, labels_g)
                generator_scores, generator_class_logits = critic(
                    generated, labels_g
                )
                generator_classification_loss = class_criterion(
                    generator_class_logits, labels_g
                )
                generator_loss = (
                    -args.wasserstein_weight * generator_scores.mean()
                    + args.generator_class_weight * generator_classification_loss
                )
                generator_loss.backward()
                optimizer_g.step()
                update_ema(ema_generator, generator, args.ema_decay)
                if hasattr(critic, "enable_hooks"):
                    critic.enable_hooks()
                set_requires_grad(critic, True)
                sums["generator"] += generator_loss.item()
                sums["generator_class"] += generator_classification_loss.item()
                sums["generator_accuracy"] += (
                    generator_class_logits.argmax(dim=1)
                    .eq(labels_g)
                    .float()
                    .mean()
                    .item()
                )
                generator_updates += 1

            progress.set_postfix(
                loss_c=f"{critic_loss.item():.3f}",
                w=f"{wasserstein_estimate.item():.3f}",
            )

        if critic_updates == 0 or generator_updates == 0:
            raise RuntimeError("The data loader produced too few non-empty batches")
        epsilon = (
            privacy_engine.get_epsilon(args.delta)
            if privacy_engine is not None
            else float("inf")
        )
        row = EpochMetrics(
            epoch=epoch,
            critic_loss=sums["critic"] / critic_updates,
            critic_classification_loss=sums["critic_class"] / critic_updates,
            critic_real_accuracy=sums["critic_accuracy"] / critic_updates,
            generator_loss=sums["generator"] / generator_updates,
            generator_classification_loss=(
                sums["generator_class"] / generator_updates
            ),
            generator_internal_accuracy=(
                sums["generator_accuracy"] / generator_updates
            ),
            wasserstein_estimate=sums["w"] / critic_updates,
            real_score=sums["real"] / critic_updates,
            fake_score=sums["fake"] / critic_updates,
            epsilon=epsilon,
            delta=args.delta,
            noise_multiplier=actual_noise_multiplier,
            clip_boundary_fraction=sums["clip_boundary"] / critic_updates,
            critic_updates=critic_updates,
            generator_updates=generator_updates,
            duration_seconds=time.time() - epoch_started,
        )
        epoch_rows.append(row)
        save_csv(output_dir / "training_metrics.csv", epoch_rows)

        ema_generator.eval()
        with torch.no_grad():
            samples = ema_generator(fixed_noise, fixed_labels)
        utils.save_image(
            samples,
            sample_dir / f"ema_samples_epoch_{epoch:03d}.png",
            nrow=10,
            normalize=True,
            value_range=(-1, 1),
        )

        should_save = epoch % args.save_every == 0 or epoch == args.epochs
        if should_save:
            private_payload = checkpoint_payload(
                generator,
                ema_generator,
                critic,
                args,
                epoch,
                epsilon,
                actual_noise_multiplier,
                time.time() - run_started,
            )
            public_payload = {
                key: value
                for key, value in private_payload.items()
                if key != "critic_private"
            }
            torch.save(
                private_payload,
                checkpoint_dir / f"training_state_private_epoch_{epoch:03d}.pt",
            )
            torch.save(public_payload, checkpoint_dir / f"dp_wgan_epoch_{epoch:03d}.pt")
            torch.save(public_payload, checkpoint_dir / "dp_wgan_last.pt")
        print(json.dumps(asdict(row), ensure_ascii=False))

    summary = {
        "status": "completed",
        "algorithm": "DP-ACWGAN-CP",
        "dataset": "MNIST train split, all digits 0-9",
        "dataset_size": len(dataset),
        "differential_privacy_enabled": not args.disable_dp,
        "epsilon": epoch_rows[-1].epsilon,
        "delta": args.delta,
        "noise_multiplier": actual_noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "weight_clip": args.weight_clip,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "critic_steps": args.critic_steps,
        "wasserstein_weight": args.wasserstein_weight,
        "critic_class_weight": args.critic_class_weight,
        "generator_class_weight": args.generator_class_weight,
        "ema_decay": args.ema_decay,
        "device": str(device),
        "torch_version": torch.__version__,
        "total_duration_seconds": time.time() - run_started,
        "final_training_metrics": asdict(epoch_rows[-1]),
        "released_model": "checkpoints/dp_wgan_last.pt (contains generator weights only)",
        "private_training_state": (
            "checkpoints/training_state_private_epoch_*.pt contains the Critic "
            "and must not be released"
        ),
        "final_sample": f"samples/ema_samples_epoch_{args.epochs:03d}.png",
        "privacy_note": "Educational reproducibility run uses secure_mode=False.",
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
