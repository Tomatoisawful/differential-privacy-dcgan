"""Train a conditional convolutional VAE with DP-SGD on MNIST."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from models import ConditionalConvVAE, initialize_weights


class VAELoss(nn.Module):
    """Per-example VAE loss compatible with Opacus ghost clipping."""

    def __init__(self, beta: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        reconstruction_logits, mu, logvar, prior_mean = outputs
        reconstruction = F.binary_cross_entropy_with_logits(
            reconstruction_logits, targets, reduction="none"
        ).flatten(1).mean(dim=1)
        kl = -0.5 * (
            1.0 + logvar - (mu - prior_mean).square() - logvar.exp()
        ).mean(dim=1)
        losses = reconstruction + self.beta * kl
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP conditional convolutional VAE on MNIST")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-digit", type=int, choices=range(10), default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-physical-batch-size", type=int, default=32)
    parser.add_argument("--grad-sample-mode", choices=("ghost", "hooks"), default="ghost")
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--label-embedding-size", type=int, default=32)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--prior-scale", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--kl-warmup-epochs", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an epoch checkpoint produced by this script.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataset(root: Path, target_digit: int | None, max_samples: int | None, seed: int):
    dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]),
    )
    indices = (
        torch.arange(len(dataset)).tolist()
        if target_digit is None
        else torch.where(dataset.targets == target_digit)[0].tolist()
    )
    if max_samples is not None:
        order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
        indices = [indices[index] for index in order[:max_samples].tolist()]
    return Subset(dataset, indices)


def unwrap(module: nn.Module) -> nn.Module:
    return getattr(module, "_module", module)


def save_resume_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: optim.Optimizer,
    privacy_engine: PrivacyEngine,
    rows: list[dict[str, float | int]],
    args: argparse.Namespace,
) -> None:
    """Atomically save all state needed to resume after a runtime interruption."""
    payload = {
        "epoch": epoch,
        "model": unwrap(model).state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "accountant": privacy_engine.accountant.state_dict(),
        "rows": rows,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    source = unwrap(source)
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if min(args.epochs, args.batch_size, args.max_physical_batch_size, args.latent_size) < 1:
        raise ValueError("epochs, batch sizes and latent size must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema decay must be in [0, 1)")

    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args.data_root, args.target_digit, args.max_samples, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = ConditionalConvVAE(
        latent_size=args.latent_size,
        label_embedding_size=args.label_embedding_size,
        features=args.features,
        prior_scale=args.prior_scale,
    ).to(device)
    model.apply(initialize_weights)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise RuntimeError(f"VAE is not Opacus-compatible: {errors}")
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    criterion = VAELoss()
    private_objects = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=loader,
        target_epsilon=args.target_epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        grad_sample_mode=args.grad_sample_mode,
    )
    if args.grad_sample_mode == "ghost":
        model, optimizer, private_criterion, loader = private_objects
    else:
        model, optimizer, loader = private_objects
        private_criterion = criterion
    noise_multiplier = float(optimizer.noise_multiplier)

    rows: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.resume is not None:
        resume_path = args.resume.resolve()
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(resume_state["model"])
        ema_model.load_state_dict(resume_state["ema_model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        privacy_engine.accountant.load_state_dict(resume_state["accountant"])
        rows = list(resume_state["rows"])
        start_epoch = int(resume_state["epoch"]) + 1
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        if start_epoch > args.epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but --epochs is {args.epochs}"
            )
    started = time.time()
    dataset_name = "MNIST digit 8" if args.target_digit == 8 else "MNIST digits 0-9"
    print(
        json.dumps(
            {
                "algorithm": "DP-CVAE",
                "dataset": dataset_name,
                "dataset_size": len(dataset),
                "device": str(device),
                "noise_multiplier": noise_multiplier,
            },
            ensure_ascii=False,
        )
    )

    resume_checkpoint_path = output_dir / "training_checkpoint.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        beta = args.beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "count": 0}
        if args.grad_sample_mode == "ghost":
            private_criterion.criterion.beta = beta
        else:
            private_criterion.beta = beta
        if args.grad_sample_mode == "hooks":
            loader_context = BatchMemoryManager(
                data_loader=loader,
                max_physical_batch_size=args.max_physical_batch_size,
                optimizer=optimizer,
            )
        else:
            from contextlib import nullcontext

            loader_context = nullcontext(loader)
        with loader_context as memory_safe_loader:
            progress = tqdm(memory_safe_loader, desc=f"vae {epoch}/{args.epochs}", unit="batch")
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                reconstruction_logits, mu, logvar, prior_mean = model(images, labels)
                reconstruction = F.binary_cross_entropy_with_logits(
                    reconstruction_logits, images, reduction="none"
                ).flatten(1).mean(dim=1)
                kl = -0.5 * (
                    1.0 + logvar - (mu - prior_mean).square() - logvar.exp()
                ).mean(dim=1)
                per_sample_loss = reconstruction + beta * kl
                loss = private_criterion(
                    (reconstruction_logits, mu, logvar, prior_mean), images
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if not getattr(optimizer, "_is_last_step_skipped", False):
                    update_ema(ema_model, model, args.ema_decay)

                batch_count = images.size(0)
                totals["loss"] += float(per_sample_loss.detach().sum())
                totals["reconstruction"] += float(reconstruction.detach().sum())
                totals["kl"] += float(kl.detach().sum())
                totals["count"] += batch_count
                progress.set_postfix(loss=f"{loss.item():.4f}", beta=f"{beta:.4f}")

        epsilon = privacy_engine.get_epsilon(args.delta)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / totals["count"],
            "reconstruction_loss": totals["reconstruction"] / totals["count"],
            "kl_loss": totals["kl"] / totals["count"],
            "beta": beta,
            "epsilon": epsilon,
            "duration_seconds": time.time() - epoch_started,
        }
        rows.append(row)
        with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        save_resume_checkpoint(
            resume_checkpoint_path,
            epoch=epoch,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            privacy_engine=privacy_engine,
            rows=rows,
            args=args,
        )
        print(json.dumps(row, ensure_ascii=False))

    ema_model.eval()
    if args.target_digit is None:
        fixed_labels = torch.arange(10, device=device).repeat_interleave(10)
    else:
        fixed_labels = torch.full((100,), args.target_digit, device=device, dtype=torch.long)
    fixed_noise = torch.randn(100, args.latent_size, device=device)
    with torch.no_grad():
        samples = ema_model.generate(fixed_noise, fixed_labels)
    utils.save_image(
        samples,
        output_dir / "generated.png",
        nrow=10,
        normalize=True,
        value_range=(-1, 1),
    )

    duration = time.time() - started
    checkpoint = {
        "algorithm": "DP-CVAE",
        "ema_model": ema_model.state_dict(),
        "model_config": {
            "latent_size": args.latent_size,
            "label_embedding_size": args.label_embedding_size,
            "features": args.features,
            "num_classes": 10,
            "image_size": 32,
            "prior_scale": args.prior_scale,
        },
        "training_config": vars(args),
        "target_digit": args.target_digit,
        "dataset_size": len(dataset),
        "privacy": {
            "enabled": True,
            "epsilon": rows[-1]["epsilon"],
            "delta": args.delta,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": args.max_grad_norm,
            "accountant": "RDP",
        },
        "total_duration_seconds": duration,
    }
    torch.save(checkpoint, output_dir / "dp_vae_final.pt")
    summary = {
        "status": "completed",
        "algorithm": checkpoint["algorithm"],
        "dataset": dataset_name,
        "dataset_size": len(dataset),
        "target_digit": args.target_digit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_size": args.latent_size,
        "beta": args.beta,
        "epsilon": rows[-1]["epsilon"],
        "delta": args.delta,
        "noise_multiplier": noise_multiplier,
        "total_duration_seconds": duration,
        "final_training_metrics": rows[-1],
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    resume_checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
