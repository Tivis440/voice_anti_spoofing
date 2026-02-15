"""
Dataset для Voice Anti-Spoofing.
Загружает wav, нарезает случайный сегмент, генерирует log-Mel на лету.
Формат разметки: CSV с колонками path, label (0=Real, 1=Fake);
при расширении можно добавить generator_id для unseen generator теста.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import librosa

from .features import compute_log_mel_tensor


class VoiceDataset(Dataset):
    """
    Dataset: по пути к wav возвращает тензор log-Mel спектрограммы и метку.
    - path: путь к wav (относительно root или абсолютный)
    - label: 0 = Real, 1 = Fake
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str = None,
        sample_rate: int = 16000,
        segment_length: float = 2.0,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        train: bool = True,
    ):
        """
        Args:
            csv_path: путь к CSV с колонками path, label
            root_dir: корень для путей из CSV (если path относительный)
            sample_rate, segment_length, n_mels, n_fft, hop_length: параметры аудио и спектрограммы
            train: если True — случайный сегмент; если False — центр файла (для валидации/теста)
        """
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir).resolve() if root_dir else Path(".").resolve()
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.train = train

        self._segment_samples = int(sample_rate * segment_length)

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        full_path = self.root_dir / path
        if not full_path.exists():
            full_path = Path(path)
        y, sr = librosa.load(str(full_path), sr=self.sample_rate, mono=True)
        return y.astype(np.float32)

    def _crop_segment(self, y: np.ndarray) -> np.ndarray:
        length = len(y)
        if length >= self._segment_samples:
            if self.train:
                start = np.random.randint(0, length - self._segment_samples + 1)
            else:
                start = (length - self._segment_samples) // 2
            return y[start : start + self._segment_samples]
        # паддинг нулями если короче
        pad = np.zeros(self._segment_samples - length, dtype=np.float32)
        return np.concatenate([y, pad])

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["label"])

        y = self._load_audio(path)
        y = self._crop_segment(y)

        spec = compute_log_mel_tensor(
            y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        return spec, label
