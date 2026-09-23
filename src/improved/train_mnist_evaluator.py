"""Train the non-private LeNet evaluator used only for DP-WGAN evaluation.

For a real private dataset, replace this model with a classifier trained on
separate public data. It is not part of the released private generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dp_wgan_models import ConvNetEvaluator, LeNetEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LeNet MNIST evaluator")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/evaluator")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--architecture", choices=("lenet", "convnet"), default="lenet"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(root: Path, train: bool, batch_size: int, workers: int) -> DataLoader:
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args.data_root, True, args.batch_size, args.workers)
    test_loader = make_loader(args.data_root, False, args.batch_size, args.workers)

    model = (
        LeNetEvaluator() if args.architecture == "lenet" else ConvNetEvaluator()
    ).to(device)
    checkpoint_name = f"{args.architecture}_mnist_best.pt"
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    rows: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(train_loader, desc=f"evaluator {epoch}/{args.epochs}", unit="batch")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            example_count += labels.numel()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        test_accuracy = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / example_count,
            "test_accuracy": test_accuracy,
        }
        rows.append(row)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "test_accuracy": test_accuracy,
                    "epoch": epoch,
                    "normalization": "Resize(32), ToTensor(), Normalize(0.5, 0.5)",
                    "feature_dimension": 84 if args.architecture == "lenet" else 128,
                    "architecture": args.architecture,
                },
                output_dir / checkpoint_name,
            )
        print(json.dumps(row, ensure_ascii=False))

    with (output_dir / "training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "best_test_accuracy": best_accuracy,
        "epochs": args.epochs,
        "duration_seconds": time.time() - started,
        "architecture": args.architecture,
        "checkpoint": checkpoint_name,
        "usage": "Evaluation only; do not release as part of the DP mechanism.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
