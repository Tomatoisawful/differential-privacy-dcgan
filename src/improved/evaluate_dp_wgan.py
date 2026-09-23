"""Generate samples and evaluate a trained DP-CWGAN-CP checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from dp_wgan_models import ConditionalGenerator, ConvNetEvaluator, LeNetEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DP-WGAN MNIST samples")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/improved/checkpoints/dp_wgan_last.pt"),
    )
    parser.add_argument(
        "--evaluator-checkpoint",
        type=Path,
        default=Path("runs/evaluator/lenet_mnist_best.pt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/evaluation")
    )
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--num-real", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--raw-generator",
        action="store_true",
        help="Evaluate raw generator weights instead of the recommended EMA weights",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(features.size(0) - 1, 1)


def symmetric_psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def feature_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """Numerically stable Fréchet distance in LeNet feature space."""
    real_features = real_features.double()
    fake_features = fake_features.double()
    real_mean = real_features.mean(dim=0)
    fake_mean = fake_features.mean(dim=0)
    real_cov = covariance(real_features)
    fake_cov = covariance(fake_features)
    real_sqrt = symmetric_psd_sqrt(real_cov)
    middle_sqrt = symmetric_psd_sqrt(real_sqrt @ fake_cov @ real_sqrt)
    mean_distance = (real_mean - fake_mean).pow(2).sum()
    fid = mean_distance + torch.trace(real_cov + fake_cov - 2.0 * middle_sqrt)
    return max(float(fid.item()), 0.0)


@torch.no_grad()
def generate_and_classify(
    generator: ConditionalGenerator,
    evaluator: nn.Module,
    num_samples: int,
    batch_size: int,
    latent_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generated_batches = []
    requested_batches = []
    prediction_batches = []
    confidence_batches = []
    feature_batches = []
    generator.eval()
    evaluator.eval()

    produced = 0
    progress = tqdm(total=num_samples, desc="generating", unit="image")
    while produced < num_samples:
        current = min(batch_size, num_samples - produced)
        # Balanced requested labels make class-consistency comparisons fair.
        requested = torch.arange(produced, produced + current, device=device) % 10
        noise = torch.randn(current, latent_size, device=device)
        images = generator(noise, requested)
        logits, features = evaluator(images, return_features=True)
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)

        generated_batches.append(images.cpu())
        requested_batches.append(requested.cpu())
        prediction_batches.append(predictions.cpu())
        confidence_batches.append(confidence.cpu())
        feature_batches.append(features.cpu())
        produced += current
        progress.update(current)
    progress.close()
    return (
        torch.cat(generated_batches),
        torch.cat(requested_batches),
        torch.cat(prediction_batches),
        torch.cat(confidence_batches),
        torch.cat(feature_batches),
    )


@torch.no_grad()
def extract_real_features(
    evaluator: nn.Module,
    data_root: Path,
    num_real: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    correct = 0
    seen = 0
    evaluator.eval()
    for images, labels in tqdm(loader, desc="real features", unit="batch"):
        remaining = num_real - seen
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        logits, batch_features = evaluator(images, return_features=True)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        seen += labels.numel()
        features.append(batch_features.cpu())
    if seen < 2:
        raise RuntimeError("At least two real examples are required for FID")
    return torch.cat(features), correct / seen


def main() -> None:
    args = parse_args()
    if min(args.num_samples, args.num_real, args.batch_size) < 2:
        raise ValueError("num-samples, num-real and batch-size must be at least 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    generator = ConditionalGenerator(
        latent_size=config["latent_size"],
        label_embedding_size=config["label_embedding_size"],
        features=config["generator_features"],
        num_classes=config.get("num_classes", 10),
    ).to(device)
    weight_key = "generator" if args.raw_generator else "ema_generator"
    generator.load_state_dict(checkpoint[weight_key])

    evaluator_checkpoint = torch.load(
        args.evaluator_checkpoint, map_location="cpu", weights_only=False
    )
    evaluator_architecture = evaluator_checkpoint.get("architecture", "lenet")
    evaluator = (
        ConvNetEvaluator()
        if evaluator_architecture == "convnet"
        else LeNetEvaluator()
    ).to(device)
    evaluator.load_state_dict(evaluator_checkpoint["model"])

    images, requested, predictions, confidence, fake_features = generate_and_classify(
        generator,
        evaluator,
        args.num_samples,
        args.batch_size,
        config["latent_size"],
        device,
    )
    real_features, real_classifier_accuracy = extract_real_features(
        evaluator,
        args.data_root,
        min(args.num_real, 10000),
        args.batch_size,
        args.workers,
        device,
    )
    fid_count = min(real_features.size(0), fake_features.size(0))
    lenet_fid = feature_fid(real_features[:fid_count], fake_features[:fid_count])

    counts = torch.bincount(predictions, minlength=10)
    probabilities = counts.double() / counts.sum()
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    normalized_entropy = entropy / math.log(10)
    class_consistency = float(predictions.eq(requested).float().mean())
    class_coverage = int(counts.gt(0).sum())
    robust_coverage = int(probabilities.ge(0.01).sum())

    grid_count = min(100, images.size(0))
    utils.save_image(
        images[:grid_count],
        output_dir / "generated_grid.png",
        nrow=10,
        normalize=True,
        value_range=(-1, 1),
    )
    uint8_images = ((images.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    torch.save(
        {
            "images_uint8": uint8_images,
            "requested_labels": requested,
            "predicted_labels": predictions,
            "classifier_confidence": confidence,
            "value_range": [0, 255],
        },
        output_dir / "generated_samples.pt",
    )

    with (output_dir / "class_histogram.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["digit", "count", "fraction"]
        )
        writer.writeheader()
        for digit in range(10):
            writer.writerow(
                {
                    "digit": digit,
                    "count": int(counts[digit]),
                    "fraction": float(probabilities[digit]),
                }
            )

    privacy = checkpoint.get("privacy", {})
    metrics = {
        "algorithm": checkpoint.get("algorithm", "DP-CWGAN-CP"),
        "generator_weights": weight_key,
        "num_generated_samples": args.num_samples,
        "num_real_samples_for_fid": fid_count,
        "class_consistency_accuracy": class_consistency,
        "mean_classifier_confidence": float(confidence.mean()),
        "class_coverage_count": class_coverage,
        "class_coverage_out_of": 10,
        "robust_class_coverage_at_1_percent": robust_coverage,
        "class_distribution_entropy": entropy,
        "normalized_class_distribution_entropy": normalized_entropy,
        "feature_fid": lenet_fid,
        "feature_fid_evaluator": evaluator_architecture,
        # Kept for compatibility with the original experiment output schema.
        "lenet_feature_fid": lenet_fid if evaluator_architecture == "lenet" else None,
        "real_test_classifier_accuracy": real_classifier_accuracy,
        "evaluator_architecture": evaluator_architecture,
        "epsilon": privacy.get("epsilon"),
        "delta": privacy.get("delta"),
        "noise_multiplier": privacy.get("noise_multiplier"),
        "max_grad_norm": privacy.get("max_grad_norm"),
        "training_duration_seconds": checkpoint.get("total_duration_seconds"),
        "evaluation_duration_seconds": time.time() - started,
        "seed": args.seed,
        "metric_note": (
            "Class consistency and feature-space FID are proxy metrics. Compare "
            "runs only with the same evaluator, sample count, preprocessing and "
            "privacy budget."
        ),
    }
    with (output_dir / "evaluation_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
