"""Инференс для модели на признаках статьи (требуется эталонный real-файл)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .features import extract_article_features, load_audio
from .model import ArticleBinaryClassifier


class InferenceError(RuntimeError):
    pass


def _build_feature_vector(sample_path: str, ref_path: str, cfg: dict) -> Tuple[np.ndarray, Dict[str, float]]:
    audio_cfg = cfg["audio"]
    feature_cfg = cfg["features"]

    y = load_audio(sample_path, audio_cfg["sample_rate"])
    ref = load_audio(ref_path, audio_cfg["sample_rate"])

    feats = extract_article_features(
        y=y,
        ref=ref,
        sr=audio_cfg["sample_rate"],
        n_mfcc=feature_cfg["n_mfcc"],
        n_fft=feature_cfg["n_fft"],
        hop_length=feature_cfg["hop_length"],
        segment_seconds=audio_cfg["segment_seconds"],
    )
    x = np.asarray([[feats[k] for k in feats.keys()]], dtype=np.float32)
    return x, feats


def predict_from_wav(sample_path: str, ref_path: str, model_path: str) -> Tuple[int, float, float, Dict[str, float]]:
    if not Path(sample_path).exists():
        raise InferenceError(f"Файл не найден: {sample_path}")
    if not Path(ref_path).exists():
        raise InferenceError(f"Эталон не найден: {ref_path}")
    if not Path(model_path).exists():
        raise InferenceError(f"Модель не найдена: {model_path}")

    payload = ArticleBinaryClassifier.load(model_path)
    pipeline = payload.get("pipeline")
    cfg = payload.get("config")

    if pipeline is None or cfg is None:
        raise InferenceError("Некорректный файл модели: отсутствует pipeline/config")

    x, feat_map = _build_feature_vector(sample_path, ref_path, cfg)
    probs = pipeline.predict_proba(x)[0]
    prob_real, prob_fake = float(probs[0]), float(probs[1])
    label = int(prob_fake >= prob_real)
    return label, prob_real, prob_fake, feat_map
