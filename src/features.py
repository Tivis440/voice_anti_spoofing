"""
Извлечение признаков для Voice Anti-Spoofing.
Log-Mel спектрограмма — базовый вход для CNN; при необходимости можно добавить
другие фичи (LFCC, CQCC) для unseen generator тестов.
"""

import numpy as np
import torch
import librosa


def compute_log_mel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    fmin: float = 0.0,
    fmax: float = None,
) -> np.ndarray:
    """
    Строит нормализованную log-Mel спектрограмму из аудиосигнала.

    Args:
        y: одномерный аудиосигнал (float)
        sr: sample rate
        n_mels: число Mel-фильтров
        n_fft: размер FFT
        hop_length: шаг окна
        fmin, fmax: диапазон частот для Mel (fmax=None → sr/2)

    Returns:
        spectrogram: shape (n_mels, time), float32, нормализовано (mean=0, std=1 по кадрам)
    """
    if fmax is None:
        fmax = float(sr) / 2.0

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    # log(1 + x) для стабильности и динамического диапазона
    log_mel = np.log1p(mel_spec).astype(np.float32)

    # Нормализация: по времени (по кадрам) для каждого мела
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    log_mel = (log_mel - mean) / std

    return log_mel


def compute_log_mel_tensor(
    y: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    **kwargs,
) -> torch.Tensor:
    """
    То же, что compute_log_mel, но возвращает torch.Tensor (1, n_mels, time)
    для подачи в модель. Удобно использовать из Dataset.
    """
    log_mel = compute_log_mel(
        y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, **kwargs
    )
    # (n_mels, time) -> (1, n_mels, time)
    return torch.from_numpy(log_mel).unsqueeze(0)
