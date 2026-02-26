"""Обучение модели на признаках статьи (без log-Mel)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .dataset import FeatureConfig, ReferenceResolver, build_feature_matrix, load_split
from .model import ArticleBinaryClassifier


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train article-based anti-spoof model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_config(root / args.config)

    logs_dir = (root / cfg["logs_dir"]).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_csv = (root / cfg["data"]["train_csv"]).resolve()
    val_csv = (root / cfg["data"]["val_csv"]).resolve()
    real_dir = (root / cfg["data"]["real_dir"]).resolve()

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
    resolver = ReferenceResolver(root, real_dir)

    print("Building train feature matrix...")
    x_train, y_train, feature_names = build_feature_matrix(df_train, resolver, feature_cfg)

    print("Building val feature matrix...")
    x_val, y_val, _ = build_feature_matrix(df_val, resolver, feature_cfg)

    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}")

    model = ArticleBinaryClassifier(
        random_state=cfg["training"].get("random_state", 42),
        max_iter=cfg["training"].get("max_iter", 2000),
    )
    model.fit(x_train, y_train)

    result = model.evaluate(x_val, y_val)
    print(f"Validation accuracy: {result.val_accuracy:.4f}")
    print(result.report)

    payload = {
        "pipeline": model.pipeline,
        "feature_names": feature_names,
        "config": cfg,
    }
    model_path = logs_dir / "best_model.pkl"
    model.save(model_path, payload)
    print(f"Saved model: {model_path}")

    metrics_path = logs_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "val_accuracy": result.val_accuracy,
                "feature_names": feature_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
