"""Обучение мультиклассовой MLP на признаках из статьи (без log-Mel)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import FeatureConfig, build_feature_matrix, load_split
from .model import FeatureMLP


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def standardize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std
    x_test_n = None if x_test is None else (x_test - mean) / std
    return x_train_n, x_val_n, x_test_n, mean.squeeze(0), std.squeeze(0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, f1, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP for voice anti-spoofing")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_config(root / args.config)

    logs_dir = (root / cfg["logs_dir"]).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_csv = (root / cfg["data"]["train_csv"]).resolve()
    val_csv = (root / cfg["data"]["val_csv"]).resolve()
    test_csv = (root / cfg["data"]["test_csv"]).resolve()

    feature_cfg = FeatureConfig(
        sample_rate=cfg["audio"]["sample_rate"],
        segment_seconds=cfg["audio"]["segment_seconds"],
        n_mfcc=cfg["features"]["n_mfcc"],
        n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"],
    )

    print("Loading splits...")
    df_train = load_split(train_csv)
    df_val = load_split(val_csv)
    df_test = load_split(test_csv)

    print("Building train features...")
    x_train, y_train, feature_names, class_to_idx = build_feature_matrix(df_train, root, feature_cfg)

    print("Building val features...")
    x_val, y_val, _, _ = build_feature_matrix(df_val, root, feature_cfg, class_to_idx=class_to_idx)

    print("Building test features...")
    x_test, y_test, _, _ = build_feature_matrix(df_test, root, feature_cfg, class_to_idx=class_to_idx)

    x_train, x_val, x_test, mean, std = standardize(x_train, x_val, x_test)

    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    print(f"Classes: {class_names}")
    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    batch_size = int(cfg["training"].get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = get_device()
    print(f"Device: {device}")

    hidden_dim = int(cfg["training"].get("hidden_dim", 128))
    dropout = float(cfg["training"].get("dropout", 0.3))
    model = FeatureMLP(
        input_dim=x_train.shape[1],
        num_classes=len(class_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    epochs = int(cfg["training"].get("epochs", 20))
    patience = int(cfg["training"].get("early_stopping_patience", 5))
    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(1, seen)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": x_train.shape[1],
                    "class_names": class_names,
                    "feature_names": feature_names,
                    "norm_mean": mean.tolist(),
                    "norm_std": std.tolist(),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "config": cfg,
                },
                logs_dir / "best_model.pt",
            )
            print(f"  -> saved {logs_dir / 'best_model.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    ckpt = torch.load(logs_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_acc, val_f1, y_val_true, y_val_pred = evaluate(model, val_loader, device)
    test_acc, test_f1, y_test_true, y_test_pred = evaluate(model, test_loader, device)

    val_report = classification_report(y_val_true, y_val_pred, target_names=class_names, digits=4)
    test_report = classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4)

    print("\nValidation report:\n" + val_report)
    print("\nTest report:\n" + test_report)

    metrics = {
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "class_names": class_names,
    }
    (logs_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics: {logs_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
