"""
Цикл обучения и валидации Voice Anti-Spoofing.
Читает сплиты из CSV, обучает CNNClassifier.
Сохраняет чекпоинты и метрики в logs/ — готово к расширению (unseen generator test).
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from .dataset import VoiceDataset
from .model import CNNClassifier


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for spec, label in loader:
        spec = spec.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        logits = model(spec)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * spec.size(0)
        n += spec.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for spec, label in loader:
        spec = spec.to(device)
        label = label.to(device)
        logits = model(spec)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Voice Anti-Spoofing CNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root dir for paths in CSV (default: current dir)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config_path = root / args.config
    config = load_config(str(config_path))

    # Пути из конфига
    raw_dir = root / config["data"]["raw_dir"]
    splits_dir = root / config["data"]["splits_dir"]
    train_csv = root / config["data"]["train_csv"]
    val_csv = root / config["data"]["val_csv"]
    logs_dir = root / config["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset и DataLoader
    train_ds = VoiceDataset(
        csv_path=str(train_csv),
        root_dir=str(root),
        sample_rate=config["sample_rate"],
        segment_length=config["segment_length"],
        n_mels=config["n_mels"],
        n_fft=config.get("n_fft", 512),
        hop_length=config.get("hop_length", 160),
        train=True,
    )
    val_ds = VoiceDataset(
        csv_path=str(val_csv),
        root_dir=str(root),
        sample_rate=config["sample_rate"],
        segment_length=config["segment_length"],
        n_mels=config["n_mels"],
        n_fft=config.get("n_fft", 512),
        hop_length=config.get("hop_length", 160),
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = CNNClassifier(in_channels=1, n_mels=config["n_mels"], n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    epochs = config["epochs"]
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = eval_accuracy(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_accuracy={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = logs_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path}")

    # Последняя эпоха тоже сохраняем (для продолжения обучения)
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        logs_dir / "last_model.pt",
    )
    print(f"Training done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
