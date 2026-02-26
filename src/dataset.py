"""Подготовка пар (sample, reference) и матрицы признаков по CSV-сплитам."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .features import extract_article_features, load_audio


REQUIRED_COLUMNS = {"path", "label"}


@dataclass
class FeatureConfig:
    sample_rate: int
    segment_seconds: float
    n_mfcc: int
    n_fft: int
    hop_length: int


class ReferenceResolver:
    """Подбирает эталон real-аудио для строки сплита."""

    def __init__(self, root_dir: Path, real_dir: Path):
        self.root_dir = root_dir
        self.real_dir = real_dir
        self.real_by_stem = self._build_real_index()

    def _build_real_index(self) -> Dict[str, Path]:
        exts = {".wav", ".flac", ".mp3", ".opus", ".m4a"}
        mapping: Dict[str, Path] = {}
        if not self.real_dir.exists():
            return mapping
        for path in self.real_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                mapping[path.stem] = path
        return mapping

    def resolve(self, row: pd.Series) -> Path:
        sample_path = self._resolve_path(row["path"])
        label = int(row["label"])

        if label == 0:
            return sample_path

        explicit_ref = str(row.get("ref_path", "")).strip()
        if explicit_ref:
            return self._resolve_path(explicit_ref)

        candidate = self.real_by_stem.get(sample_path.stem)
        if candidate is not None:
            return candidate

        raise FileNotFoundError(
            f"Не найден эталон для fake файла: {sample_path}. "
            "Добавьте колонку ref_path в CSV или совпадающие stem в real."
        )

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.root_dir / path).resolve()


def load_split(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"В {csv_path} отсутствуют колонки: {sorted(missing)}")
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    resolver: ReferenceResolver,
    feature_cfg: FeatureConfig,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    labels = []
    feature_names: list[str] | None = None

    for idx, row in df.iterrows():
        sample_path = resolver._resolve_path(row["path"])
        ref_path = resolver.resolve(row)

        if not sample_path.exists():
            raise FileNotFoundError(f"Файл не найден: {sample_path}")
        if not ref_path.exists():
            raise FileNotFoundError(f"Эталон не найден: {ref_path}")

        y = load_audio(sample_path, feature_cfg.sample_rate)
        ref = load_audio(ref_path, feature_cfg.sample_rate)

        feats = extract_article_features(
            y=y,
            ref=ref,
            sr=feature_cfg.sample_rate,
            n_mfcc=feature_cfg.n_mfcc,
            n_fft=feature_cfg.n_fft,
            hop_length=feature_cfg.hop_length,
            segment_seconds=feature_cfg.segment_seconds,
        )

        if feature_names is None:
            feature_names = list(feats.keys())
        rows.append([feats[name] for name in feature_names])
        labels.append(int(row["label"]))

        if (idx + 1) % 200 == 0:
            print(f"  processed: {idx + 1}/{len(df)}")

    if not rows:
        raise ValueError("Пустой сплит: нет строк для обучения")

    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), feature_names or []
