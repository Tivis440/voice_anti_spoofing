"""Подготовка feature matrix для мультиклассовой классификации real/fake_engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .features import extract_article_features_single, load_audio

REQUIRED_COLUMNS = {"path"}


@dataclass
class FeatureConfig:
    sample_rate: int
    segment_seconds: float
    n_mfcc: int
    n_fft: int
    hop_length: int


def load_split(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"В {csv_path} отсутствуют колонки: {sorted(missing)}")
    return df


def resolve_path(root_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def infer_class_name(path_value: str, label_value: object | None = None) -> str:
    parts = Path(path_value).parts
    if "real" in parts:
        return "real"
    if "fake" in parts:
        idx = parts.index("fake")
        if idx + 1 < len(parts) - 1:
            engine = parts[idx + 1]
            return f"fake_{engine}"
        return "fake"
    if label_value is not None:
        try:
            numeric = int(label_value)
            return "real" if numeric == 0 else "fake"
        except Exception:  # noqa: BLE001
            return str(label_value)
    return "unknown"


def get_class_names(df: pd.DataFrame) -> list[str]:
    if "class_name" in df.columns:
        names = df["class_name"].astype(str).tolist()
    else:
        label_col = df["label"] if "label" in df.columns else [None] * len(df)
        names = [infer_class_name(p, l) for p, l in zip(df["path"], label_col)]
    uniq = sorted(set(names))
    if "real" in uniq:
        uniq.remove("real")
        uniq = ["real", *uniq]
    return uniq


def build_feature_matrix(
    df: pd.DataFrame,
    root_dir: Path,
    feature_cfg: FeatureConfig,
    class_to_idx: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[str], Dict[str, int]]:
    rows = []
    labels = []
    feature_names: list[str] | None = None

    if "class_name" in df.columns:
        class_names = df["class_name"].astype(str).tolist()
    else:
        label_col = df["label"] if "label" in df.columns else [None] * len(df)
        class_names = [infer_class_name(p, l) for p, l in zip(df["path"], label_col)]

    if class_to_idx is None:
        uniq = sorted(set(class_names))
        if "real" in uniq:
            uniq.remove("real")
            uniq = ["real", *uniq]
        class_to_idx = {name: idx for idx, name in enumerate(uniq)}

    for idx, row in df.iterrows():
        path = resolve_path(root_dir, row["path"])
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        class_name = class_names[idx]
        if class_name not in class_to_idx:
            raise ValueError(
                f"Класс '{class_name}' отсутствует в train-словаре. "
                "Проверьте соответствие train/val/test по классам."
            )

        y = load_audio(path, feature_cfg.sample_rate)
        feats = extract_article_features_single(
            y=y,
            sr=feature_cfg.sample_rate,
            n_mfcc=feature_cfg.n_mfcc,
            n_fft=feature_cfg.n_fft,
            hop_length=feature_cfg.hop_length,
            segment_seconds=feature_cfg.segment_seconds,
        )

        if feature_names is None:
            feature_names = list(feats.keys())
        rows.append([feats[name] for name in feature_names])
        labels.append(class_to_idx[class_name])

        if (idx + 1) % 200 == 0:
            print(f"  processed: {idx + 1}/{len(df)}")

    if not rows:
        raise ValueError("Пустой сплит: нет строк для обучения")

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        feature_names or [],
        class_to_idx,
    )
