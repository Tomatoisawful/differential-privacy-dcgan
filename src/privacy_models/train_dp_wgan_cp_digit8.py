"""Train an unconditional DP-WGAN-CP on MNIST digit 8."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from models import SingleDigitCritic, SingleDigitGenerator, initialize_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP-WGAN-CP on MNIST digit 8")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-digit", type=int, choices=range(10), default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--critic-features", type=int, default=32)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--weight-clip", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0.99)
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


def unwrap(module: nn.Module) -> nn.Module:
    return getattr(module, "_module", module)


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer)


@torch.no_grad()
def clip_critic(critic: nn.Module, bound: float) -> float:
    boundary = 0
    total = 0
    for parameter in unwrap(critic).parameters():
        parameter.clamp_(-bound, bound)
        boundary += parameter.abs().ge(bound * 0.999).sum().item()
        total += parameter.numel()
    return boundary / total


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def save_resume_checkpoint(
    path: Path,
    *,
    epoch: int,
    generator: nn.Module,
    critic: nn.Module,
    ema_generator: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_c: optim.Optimizer,
    privacy_engine: PrivacyEngine,
    rows: list[dict[str, float | int]],
    global_critic_steps: int,
    args: argparse.Namespace,
) -> None:
    """Atomically save all state needed to resume after a runtime interruption."""
    payload = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "critic": unwrap(critic).state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_c": optimizer_c.state_dict(),
        "accountant": privacy_engine.accountant.state_dict(),
        "rows": rows,
        "global_critic_steps": global_critic_steps,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if min(args.epochs, args.batch_size, args.critic_steps) < 1:
        raise ValueError("epochs, batch size and critic steps must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = datasets.MNIST(
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
    indices = torch.where(full_dataset.targets == args.target_digit)[0].tolist()
    if args.max_samples is not None:
        order = torch.randperm(
            len(indices), generator=torch.Generator().manual_seed(args.seed)
        )[: args.max_samples]
        indices = [indices[index] for index in order.tolist()]
    dataset = Subset(full_dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )

    generator = SingleDigitGenerator(args.latent_size, args.generator_features).to(device)
    critic = SingleDigitCritic(args.critic_features).to(device)
    generator.apply(initialize_weights)
    critic.apply(initialize_weights)
    ema_generator = copy.deepcopy(generator).eval()
    ema_generator.requires_grad_(False)
    errors = ModuleValidator.validate(critic, strict=False)
    if errors:
        raise RuntimeError(f"Critic is not Opacus-compatible: {errors}")

    optimizer_g = optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(0.0, 0.9))
    optimizer_c = optim.RMSprop(critic.parameters(), lr=args.critic_lr)
    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    critic, optimizer_c, loader = privacy_engine.make_private_with_epsilon(
        module=critic,
        optimizer=optimizer_c,
        data_loader=loader,
        target_epsilon=args.target_epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
    )
    noise_multiplier = float(optimizer_c.noise_multiplier)

    fixed_noise = torch.randn(100, args.latent_size, device=device)
    rows: list[dict[str, float | int]] = []
    started = time.time()
    global_critic_steps = 0
    start_epoch = 1
    if args.resume is not None:
        resume_path = args.resume.resolve()
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        generator.load_state_dict(resume_state["generator"])
        unwrap(critic).load_state_dict(resume_state["critic"])
        ema_generator.load_state_dict(resume_state["ema_generator"])
        optimizer_g.load_state_dict(resume_state["optimizer_g"])
        optimizer_c.load_state_dict(resume_state["optimizer_c"])
        privacy_engine.accountant.load_state_dict(resume_state["accountant"])
        rows = list(resume_state["rows"])
        global_critic_steps = int(resume_state["global_critic_steps"])
        start_epoch = int(resume_state["epoch"]) + 1
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        if start_epoch > args.epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but --epochs is {args.epochs}"
            )
    print(
        json.dumps(
            {
                "algorithm": "DP-WGAN-CP",
                "dataset": f"MNIST digit {args.target_digit}",
                "dataset_size": len(dataset),
                "noise_multiplier": noise_multiplier,
                "device": str(device),
            },
            ensure_ascii=False,
        )
    )

    resume_checkpoint_path = output_dir / "training_checkpoint.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        totals = {"critic": 0.0, "generator": 0.0, "w": 0.0, "clip": 0.0}
        critic_updates = 0
        generator_updates = 0
        progress = tqdm(loader, desc=f"wgan-cp {epoch}/{args.epochs}", unit="batch")
        for real_images, _ in progress:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            set_requires_grad(critic, True)
            optimizer_c.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_images = generator(torch.randn(batch_size, args.latent_size, device=device))
            real_scores = critic(real_images)
            fake_scores = critic(fake_images)
            critic_loss = fake_scores.mean() - real_scores.mean()
            critic_loss.backward()
            optimizer_c.step()
            boundary_fraction = clip_critic(critic, args.weight_clip)

            global_critic_steps += 1
            critic_updates += 1
            wasserstein = real_scores.mean() - fake_scores.mean()
            totals["critic"] += critic_loss.item()
            totals["w"] += wasserstein.item()
            totals["clip"] += boundary_fraction

            if global_critic_steps % args.critic_steps == 0:
                optimizer_g.zero_grad(set_to_none=True)
                set_requires_grad(critic, False)
                if hasattr(critic, "disable_hooks"):
                    critic.disable_hooks()
                generated = generator(torch.randn(batch_size, args.latent_size, device=device))
                generator_loss = -critic(generated).mean()
                generator_loss.backward()
                optimizer_g.step()
                update_ema(ema_generator, generator, args.ema_decay)
                if hasattr(critic, "enable_hooks"):
                    critic.enable_hooks()
                set_requires_grad(critic, True)
                totals["generator"] += generator_loss.item()
                generator_updates += 1

            progress.set_postfix(c=f"{critic_loss.item():.4f}", w=f"{wasserstein.item():.4f}")

        if generator_updates == 0:
            raise RuntimeError("No generator updates were performed")
        epsilon = privacy_engine.get_epsilon(args.delta)
        row = {
            "epoch": epoch,
            "critic_loss": totals["critic"] / critic_updates,
            "generator_loss": totals["generator"] / generator_updates,
            "wasserstein_estimate": totals["w"] / critic_updates,
            "clip_boundary_fraction": totals["clip"] / critic_updates,
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
            generator=generator,
            critic=critic,
            ema_generator=ema_generator,
            optimizer_g=optimizer_g,
            optimizer_c=optimizer_c,
            privacy_engine=privacy_engine,
            rows=rows,
            global_critic_steps=global_critic_steps,
            args=args,
        )
        print(json.dumps(row, ensure_ascii=False))

    ema_generator.eval()
    with torch.no_grad():
        samples = ema_generator(fixed_noise)
    utils.save_image(
        samples,
        output_dir / "generated.png",
        nrow=10,
        normalize=True,
        value_range=(-1, 1),
    )

    duration = time.time() - started
    checkpoint = {
        "algorithm": "DP-WGAN-CP",
        "ema_generator": ema_generator.state_dict(),
        "model_config": {
            "latent_size": args.latent_size,
            "generator_features": args.generator_features,
            "critic_features": args.critic_features,
            "image_size": 32,
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
    torch.save(checkpoint, output_dir / "dp_wgan_cp_digit8_final.pt")
    summary = {
        "status": "completed",
        "algorithm": checkpoint["algorithm"],
        "dataset": f"MNIST digit {args.target_digit}",
        "dataset_size": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "epsilon": rows[-1]["epsilon"],
        "delta": args.delta,
        "noise_multiplier": noise_multiplier,
        "weight_clip": args.weight_clip,
        "total_duration_seconds": duration,
        "final_training_metrics": rows[-1],
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    resume_checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
