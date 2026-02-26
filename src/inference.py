"""Инференс: один аудиофайл -> real/fake + вероятности классов."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .features import extract_article_features_single, load_audio
from .model import FeatureMLP


class InferenceError(RuntimeError):
    pass


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_from_wav(
    sample_path: str,
    model_path: str,
) -> Tuple[str, bool, Dict[str, float], Dict[str, float]]:
    """Возвращает (predicted_class, is_fake, probs_by_class, features)."""
    sample = Path(sample_path)
    model_file = Path(model_path)

    if not sample.exists():
        raise InferenceError(f"Файл не найден: {sample}")
    if not model_file.exists():
        raise InferenceError(f"Модель не найдена: {model_file}")

    device = _get_device()
    ckpt = torch.load(model_file, map_location=device, weights_only=False)

    class_names = ckpt.get("class_names")
    feature_names = ckpt.get("feature_names")
    norm_mean = np.asarray(ckpt.get("norm_mean"), dtype=np.float32)
    norm_std = np.asarray(ckpt.get("norm_std"), dtype=np.float32)
    cfg = ckpt.get("config")

    if not all([class_names, feature_names, cfg is not None]):
        raise InferenceError("Некорректный чекпоинт: отсутствуют class_names/feature_names/config")

    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]

    y = load_audio(sample, sample_rate=audio_cfg["sample_rate"])
    feature_map = extract_article_features_single(
        y=y,
        sr=audio_cfg["sample_rate"],
        n_mfcc=feat_cfg["n_mfcc"],
        n_fft=feat_cfg["n_fft"],
        hop_length=feat_cfg["hop_length"],
        segment_seconds=audio_cfg["segment_seconds"],
    )

    x = np.asarray([[feature_map[name] for name in feature_names]], dtype=np.float32)
    x = (x - norm_mean[None, :]) / np.where(norm_std[None, :] < 1e-8, 1.0, norm_std[None, :])

    hidden_dim = int(ckpt.get("hidden_dim", 128))
    dropout = float(ckpt.get("dropout", 0.3))
    model = FeatureMLP(
        input_dim=x.shape[1],
        num_classes=len(class_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    predicted_class = class_names[best_idx]
    probs_by_class = {name: float(probs[i]) for i, name in enumerate(class_names)}
    is_fake = predicted_class != "real"

    return predicted_class, is_fake, probs_by_class, feature_map
