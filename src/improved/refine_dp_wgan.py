"""Privacy-preserving post-processing for a trained conditional DP-WGAN.

This script never opens the private training split.  It refines the released
generator with a frozen *public* classifier and anchors every update to the
original DP generator.  Therefore it does not spend additional privacy budget
when the guide checkpoint is genuinely public/independent of the private data.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import utils
from tqdm import trange

from dp_wgan_models import ConditionalGenerator, LeNetEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine a released DP generator")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--guide-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--tv-weight", type=float, default=0.002)
    parser.add_argument("--augmentation-noise", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def total_variation(images: torch.Tensor) -> torch.Tensor:
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def balanced_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    repeats = (batch_size + 9) // 10
    labels = torch.arange(10, device=device).repeat(repeats)[:batch_size]
    return labels[torch.randperm(batch_size, device=device)]


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size) <= 0 or args.learning_rate <= 0:
        raise ValueError("steps, batch-size and learning-rate must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    generator = ConditionalGenerator(
        latent_size=config["latent_size"],
        label_embedding_size=config["label_embedding_size"],
        features=config["generator_features"],
        num_classes=config.get("num_classes", 10),
    ).to(device)
    generator.load_state_dict(checkpoint["ema_generator"])
    anchor_generator = copy.deepcopy(generator).eval()
    anchor_generator.requires_grad_(False)

    guide_payload = torch.load(
        args.guide_checkpoint, map_location="cpu", weights_only=False
    )
    guide = LeNetEvaluator().to(device).eval()
    guide.load_state_dict(guide_payload["model"])
    guide.requires_grad_(False)

    optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    started = time.time()
    last = {}
    generator.train()
    for step in trange(1, args.steps + 1, desc="public post-processing", unit="step"):
        labels = balanced_labels(args.batch_size, device)
        noise = torch.randn(args.batch_size, config["latent_size"], device=device)
        with torch.no_grad():
            anchors = anchor_generator(noise, labels)
        images = generator(noise, labels)

        # Classification through both clean and mildly perturbed images reduces
        # the chance of optimizing brittle, classifier-specific pixel artifacts.
        clean_logits = guide(images)
        perturbed = (images + torch.randn_like(images) * args.augmentation_noise).clamp(-1, 1)
        perturbed_logits = guide(perturbed)
        class_loss = 0.5 * (
            F.cross_entropy(clean_logits, labels)
            + F.cross_entropy(perturbed_logits, labels)
        )
        anchor_loss = F.l1_loss(images, anchors)
        tv_loss = total_variation(images)
        loss = (
            class_loss
            + args.anchor_weight * anchor_loss
            + args.tv_weight * tv_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            accuracy = clean_logits.argmax(1).eq(labels).float().mean().item()
            last = {
                "step": step,
                "loss": float(loss.item()),
                "class_loss": float(class_loss.item()),
                "anchor_l1": float(anchor_loss.item()),
                "total_variation": float(tv_loss.item()),
                "guide_accuracy": accuracy,
            }
            print(json.dumps(last, ensure_ascii=False))

    generator.eval()
    fixed_labels = torch.arange(10, device=device).repeat_interleave(10)
    fixed_noise = torch.randn(100, config["latent_size"], device=device)
    with torch.no_grad():
        samples = generator(fixed_noise, fixed_labels)
    utils.save_image(
        samples,
        output_dir / "refined_samples.png",
        nrow=10,
        normalize=True,
        value_range=(-1, 1),
    )

    result = copy.deepcopy(checkpoint)
    result["algorithm"] = "DP-ACWGAN-CP + public-classifier post-processing"
    state = generator.state_dict()
    result["generator"] = state
    result["ema_generator"] = state
    result["post_processing"] = {
        "uses_private_training_examples": False,
        "additional_privacy_cost": 0.0,
        "guide_checkpoint": str(args.guide_checkpoint),
        "guide_must_be_public_or_independent": True,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "anchor_weight": args.anchor_weight,
        "tv_weight": args.tv_weight,
        "augmentation_noise": args.augmentation_noise,
        "duration_seconds": time.time() - started,
        "final_optimization_metrics": last,
    }
    torch.save(result, output_dir / "dp_wgan_refined.pt")
    with (output_dir / "refinement_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result["post_processing"], stream, ensure_ascii=False, indent=2)
    print(json.dumps(result["post_processing"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
