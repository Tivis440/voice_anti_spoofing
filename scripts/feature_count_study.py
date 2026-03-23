from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import FeatureConfig, build_feature_matrix, load_split
from src.model import FeatureMLP
from src.train import evaluate, get_device, load_config, standardize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Исследование влияния количества признаков на качество модели"
    )
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--mfcc-values",
        type=str,
        default="5,8,13,16,20,24",
        help="Список значений n_mfcc через запятую",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="feature_count_study",
        help="Префикс выходных файлов в logs/",
    )
    return parser.parse_args()


def train_single_run(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
) -> dict[str, float]:
    x_train, x_val, x_test, _, _ = standardize(x_train, x_val, x_test)

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    batch_size = int(cfg["training"].get("batch_size", 64))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = get_device()
    hidden_dim = int(cfg["training"].get("hidden_dim", 128))
    dropout = float(cfg["training"].get("dropout", 0.3))
    model = FeatureMLP(
        input_dim=x_train.shape[1],
        num_classes=len(np.unique(y_train)),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    epochs = int(cfg["training"].get("epochs", 20))
    patience = int(cfg["training"].get("early_stopping_patience", 5))
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            seen += x_batch.size(0)

        train_loss = running_loss / max(1, seen)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(
            f"    epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("    early stopping")
                break

    if best_state is None:
        raise RuntimeError("Не удалось сохранить состояние модели в ходе эксперимента")

    model.load_state_dict(best_state)
    model = model.to(device)
    val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
    test_acc, test_f1, _, _ = evaluate(model, test_loader, device)

    return {
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
    }


def save_plot(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: matplotlib недоступен, график исследования не сохранен ({exc})")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["feature_count"], df["val_accuracy"], marker="o", label="val_accuracy")
    ax.plot(df["feature_count"], df["val_macro_f1"], marker="o", label="val_macro_f1")
    ax.plot(df["feature_count"], df["test_accuracy"], marker="o", label="test_accuracy")
    ax.plot(df["feature_count"], df["test_macro_f1"], marker="o", label="test_macro_f1")
    ax.set_xlabel("Количество признаков")
    ax.set_ylabel("Метрика")
    ax.set_title("Влияние количества признаков на качество модели")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    cfg = load_config(root / args.config)

    logs_dir = root / cfg["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_csv = root / cfg["data"]["train_csv"]
    val_csv = root / cfg["data"]["val_csv"]
    test_csv = root / cfg["data"]["test_csv"]

    df_train = load_split(train_csv)
    df_val = load_split(val_csv)
    df_test = load_split(test_csv)

    n_mfcc_values = [int(v.strip()) for v in args.mfcc_values.split(",") if v.strip()]
    results: list[dict[str, float | int]] = []

    for n_mfcc in n_mfcc_values:
        print(f"\n=== n_mfcc={n_mfcc} ===")
        feature_cfg = FeatureConfig(
            sample_rate=cfg["audio"]["sample_rate"],
            segment_seconds=cfg["audio"]["segment_seconds"],
            n_mfcc=n_mfcc,
            n_fft=cfg["features"]["n_fft"],
            hop_length=cfg["features"]["hop_length"],
        )

        print("  building train features...")
        x_train, y_train, feature_names, class_to_idx = build_feature_matrix(df_train, root, feature_cfg)
        print("  building val features...")
        x_val, y_val, _, _ = build_feature_matrix(df_val, root, feature_cfg, class_to_idx=class_to_idx)
        print("  building test features...")
        x_test, y_test, _, _ = build_feature_matrix(df_test, root, feature_cfg, class_to_idx=class_to_idx)

        metrics = train_single_run(x_train, y_train, x_val, y_val, x_test, y_test, cfg)
        metrics["n_mfcc"] = n_mfcc
        metrics["feature_count"] = len(feature_names)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(["feature_count", "n_mfcc"]).reset_index(drop=True)
    csv_path = logs_dir / f"{args.out_prefix}.csv"
    json_path = logs_dir / f"{args.out_prefix}.json"
    png_path = logs_dir / f"{args.out_prefix}.png"

    results_df.to_csv(csv_path, index=False)
    json_path.write_text(results_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    save_plot(results_df, png_path)

    print("\nРезультаты исследования:")
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
