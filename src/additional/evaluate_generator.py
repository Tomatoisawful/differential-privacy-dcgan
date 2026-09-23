"""Evaluate DP-VAE, single-digit DP-WGAN-CP and mandatory DP-DCGAN models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "improved"))
sys.path.insert(0, str(HERE.parent / "required"))

from dp_wgan_models import ConvNetEvaluator, LeNetEvaluator  # noqa: E402
from models import ConditionalConvVAE, SingleDigitGenerator  # noqa: E402
from train_dp_dcgan import Generator as DCGANGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MNIST generator")
    parser.add_argument("--model-type", choices=("dp_vae", "dp_wgan_cp", "dp_dcgan"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--evaluator-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("all", "single"), required=True)
    parser.add_argument("--target-digit", type=int, choices=range(10), default=8)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--num-real", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    return (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.T


def feature_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    real_features = real_features.double()
    fake_features = fake_features.double()
    real_mean = real_features.mean(0)
    fake_mean = fake_features.mean(0)
    real_cov = covariance(real_features)
    fake_cov = covariance(fake_features)
    real_sqrt = symmetric_psd_sqrt(real_cov)
    middle_sqrt = symmetric_psd_sqrt(real_sqrt @ fake_cov @ real_sqrt)
    value = (real_mean - fake_mean).square().sum() + torch.trace(
        real_cov + fake_cov - 2.0 * middle_sqrt
    )
    return max(float(value), 0.0)


def load_generator(args: argparse.Namespace, device: torch.device):
    if args.model_type == "dp_dcgan":
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = DCGANGenerator(latent_size=64, features=32, image_size=32).to(device)
        model.load_state_dict(state)
        metadata = {}
        if args.metadata_json is not None:
            metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        return model.eval(), 64, metadata.get("mandatory_task", "DP-SGD + DCGAN"), metadata

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    if args.model_type == "dp_vae":
        model = ConditionalConvVAE(
            latent_size=config["latent_size"],
            label_embedding_size=config["label_embedding_size"],
            features=config["features"],
            num_classes=config.get("num_classes", 10),
            prior_scale=config.get("prior_scale", 0.0),
        ).to(device)
        model.load_state_dict(checkpoint["ema_model"])
    else:
        model = SingleDigitGenerator(
            latent_size=config["latent_size"], features=config["generator_features"]
        ).to(device)
        model.load_state_dict(checkpoint["ema_generator"])
    return model.eval(), config["latent_size"], checkpoint["algorithm"], checkpoint


@torch.no_grad()
def generate_and_classify(
    model: nn.Module,
    model_type: str,
    evaluator: nn.Module,
    task: str,
    target_digit: int,
    latent_size: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
):
    image_batches = []
    requested_batches = []
    prediction_batches = []
    confidence_batches = []
    feature_batches = []
    produced = 0
    progress = tqdm(total=num_samples, desc="generating", unit="image")
    while produced < num_samples:
        current = min(batch_size, num_samples - produced)
        if task == "all":
            requested = torch.arange(produced, produced + current, device=device) % 10
        else:
            requested = torch.full((current,), target_digit, device=device, dtype=torch.long)
        if model_type == "dp_dcgan":
            images = model(torch.randn(current, latent_size, 1, 1, device=device))
        elif model_type == "dp_vae":
            images = model.generate(torch.randn(current, latent_size, device=device), requested)
        else:
            images = model(torch.randn(current, latent_size, device=device))
        logits, features = evaluator(images, return_features=True)
        probabilities = logits.softmax(1)
        confidence, predictions = probabilities.max(1)
        image_batches.append(images.cpu())
        requested_batches.append(requested.cpu())
        prediction_batches.append(predictions.cpu())
        confidence_batches.append(confidence.cpu())
        feature_batches.append(features.cpu())
        produced += current
        progress.update(current)
    progress.close()
    return tuple(
        torch.cat(items)
        for items in (
            image_batches,
            requested_batches,
            prediction_batches,
            confidence_batches,
            feature_batches,
        )
    )


@torch.no_grad()
def extract_real_features(
    evaluator: nn.Module,
    data_root: Path,
    task: str,
    target_digit: int,
    num_real: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    feature_batches = []
    correct = 0
    seen = 0
    for images, labels in tqdm(loader, desc="real features", unit="batch"):
        if task == "single":
            selected = labels.eq(target_digit)
            images = images[selected]
            labels = labels[selected]
        remaining = num_real - seen
        if remaining <= 0:
            break
        images = images[:remaining]
        labels = labels[:remaining]
        if images.numel() == 0:
            continue
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, features = evaluator(images, return_features=True)
        correct += logits.argmax(1).eq(labels).sum().item()
        seen += labels.numel()
        feature_batches.append(features.cpu())
    if seen < 2:
        raise RuntimeError("At least two real reference images are required")
    return torch.cat(feature_batches), correct / seen


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    model, latent_size, algorithm, metadata = load_generator(args, device)
    evaluator_payload = torch.load(args.evaluator_checkpoint, map_location="cpu", weights_only=False)
    architecture = evaluator_payload.get("architecture", "lenet")
    evaluator = (ConvNetEvaluator() if architecture == "convnet" else LeNetEvaluator()).to(device)
    evaluator.load_state_dict(evaluator_payload["model"])
    evaluator.eval()

    images, requested, predictions, confidence, fake_features = generate_and_classify(
        model,
        args.model_type,
        evaluator,
        args.task,
        args.target_digit,
        latent_size,
        args.num_samples,
        args.batch_size,
        device,
    )
    real_features, real_accuracy = extract_real_features(
        evaluator,
        args.data_root,
        args.task,
        args.target_digit,
        args.num_real,
        args.batch_size,
        args.workers,
        device,
    )
    fid_count = min(real_features.size(0), fake_features.size(0))
    fid = feature_fid(real_features[:fid_count], fake_features[:fid_count])
    counts = torch.bincount(predictions, minlength=10)
    fractions = counts.double() / counts.sum()
    nonzero = fractions > 0
    entropy = float(-(fractions[nonzero] * fractions[nonzero].log()).sum())
    requested_accuracy = float(predictions.eq(requested).float().mean())

    utils.save_image(
        images[:100],
        output_dir / "generated_grid.png",
        nrow=10,
        normalize=True,
        value_range=(-1, 1),
    )
    with (output_dir / "class_histogram.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("digit", "count", "fraction"))
        writer.writeheader()
        for digit in range(10):
            writer.writerow(
                {"digit": digit, "count": int(counts[digit]), "fraction": float(fractions[digit])}
            )

    privacy = metadata.get("privacy", {})
    if args.model_type == "dp_dcgan":
        privacy = {
            "epsilon": metadata.get("epsilon"),
            "delta": metadata.get("delta"),
            "noise_multiplier": metadata.get("noise_multiplier"),
            "max_grad_norm": metadata.get("max_grad_norm"),
        }
    metrics = {
        "algorithm": algorithm,
        "task": "MNIST 0-9" if args.task == "all" else f"MNIST digit {args.target_digit}",
        "evaluator_architecture": architecture,
        "num_generated_samples": args.num_samples,
        "num_real_samples_for_fid": fid_count,
        "requested_label_accuracy": requested_accuracy,
        "class_consistency_accuracy": requested_accuracy if args.task == "all" else None,
        "target_digit_accuracy": requested_accuracy if args.task == "single" else None,
        "mean_classifier_confidence": float(confidence.mean()),
        "class_coverage_count": int(counts.gt(0).sum()),
        "robust_class_coverage_at_1_percent": int(fractions.ge(0.01).sum()),
        "class_distribution_entropy": entropy,
        "normalized_class_distribution_entropy": entropy / math.log(10),
        "feature_fid": fid,
        "fid_reference": "MNIST test 0-9" if args.task == "all" else f"MNIST test digit {args.target_digit}",
        "real_reference_classifier_accuracy": real_accuracy,
        "epsilon": privacy.get("epsilon"),
        "delta": privacy.get("delta"),
        "noise_multiplier": privacy.get("noise_multiplier"),
        "max_grad_norm": privacy.get("max_grad_norm"),
        "training_duration_seconds": metadata.get(
            "total_duration_seconds", metadata.get("total_duration_seconds")
        ),
        "evaluation_duration_seconds": time.time() - started,
        "seed": args.seed,
        "metric_note": (
            "Compare feature FID only within the same task, evaluator, sample count and preprocessing. "
            "For a single-digit model, requested-label accuracy is the target-digit recognition rate."
        ),
    }
    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
