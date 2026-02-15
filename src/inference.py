"""
Инференс для Voice Anti-Spoofing: загрузка модели и предсказание по wav-файлу.
Используется GUI и может вызываться из CLI.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
import librosa

from .model import CNNClassifier
from .features import compute_log_mel_tensor


def load_model_and_config(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[CNNClassifier, dict]:
    """Загружает чекпоинт (с конфигом внутри) и возвращает модель + конфиг."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = ckpt.get("config")
    if config is None:
        config_path = Path(checkpoint_path).resolve().parent.parent / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    model = CNNClassifier(
        in_channels=1,
        n_mels=config["n_mels"],
        n_classes=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, config


def predict_from_wav(
    wav_path: str,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    config_overrides: Optional[dict] = None,
) -> Tuple[int, float, float]:
    """
    Классифицирует один wav-файл: Real (0) или Fake (1).

    Returns:
        label: 0 = Real, 1 = Fake
        prob_real: вероятность Real
        prob_fake: вероятность Fake
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_and_config(checkpoint_path, device)
    if config_overrides:
        config = {**config, **config_overrides}

    sr = config["sample_rate"]
    segment_length = config["segment_length"]
    n_mels = config["n_mels"]
    n_fft = config.get("n_fft", 512)
    hop_length = config.get("hop_length", 160)

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    segment_samples = int(sr * segment_length)
    if len(y) >= segment_samples:
        start = (len(y) - segment_samples) // 2
        y = y[start : start + segment_samples]
    else:
        import numpy as np
        pad = np.zeros(segment_samples - len(y), dtype=y.dtype)
        y = np.concatenate([y, pad])

    spec = compute_log_mel_tensor(
        y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_real = float(probs[0])
    prob_fake = float(probs[1])
    label = 1 if prob_fake >= prob_real else 0
    return label, prob_real, prob_fake
