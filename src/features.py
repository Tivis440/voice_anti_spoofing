"""Признаки по статье для детекции синтезированного голоса (без log-Mel)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Dict

import audioread
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly, stft

EPS = 1e-10


def _read_via_audioread(path: str | Path) -> tuple[np.ndarray, int]:
    chunks = []
    sample_rate = 16000
    channels = 1
    with audioread.audio_open(str(path)) as f:
        sample_rate = f.samplerate
        channels = f.channels
        for chunk in f:
            chunks.append(np.frombuffer(chunk, dtype=np.int16))
    if not chunks:
        return np.zeros(1, dtype=np.float32), sample_rate
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Загружает mono-аудио и приводит к sample_rate."""
    try:
        y, src_sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
    except Exception:  # noqa: BLE001
        y, src_sr = _read_via_audioread(path)

    if src_sr != sample_rate:
        ratio = Fraction(sample_rate, src_sr).limit_denominator()
        y = resample_poly(y, ratio.numerator, ratio.denominator).astype(np.float32)
    return y


def _center_crop_or_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        return y[start : start + target_len]
    pad = np.zeros(target_len - len(y), dtype=np.float32)
    return np.concatenate([y, pad])


def _align_pair(y: np.ndarray, ref: np.ndarray, segment_samples: int) -> tuple[np.ndarray, np.ndarray]:
    y = _center_crop_or_pad(y, segment_samples)
    ref = _center_crop_or_pad(ref, segment_samples)
    return y, ref


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmin, fmax = 0.0, sr / 2.0
    mels = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def _mfcc(y: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    _, _, zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, padded=False)
    power = (np.abs(zxx) ** 2).astype(np.float32)
    mel_fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=max(26, n_mfcc * 2))
    mel_spec = np.dot(mel_fb, power)
    log_mel = np.log(mel_spec + EPS)
    coeffs = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return coeffs.astype(np.float32)


def _framewise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    min_t = min(a.shape[1], b.shape[1])
    if min_t == 0:
        return np.array([0.0], dtype=np.float32)
    diff = a[:, :min_t] - b[:, :min_t]
    return np.sqrt(np.sum(diff * diff, axis=0, dtype=np.float64)).astype(np.float32)


def _spectral_centroid(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    _, _, zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, padded=False)
    mag = np.abs(zxx)
    freqs = np.linspace(0, sr / 2, mag.shape[0], dtype=np.float32)[:, None]
    denom = np.sum(mag, axis=0, dtype=np.float64) + EPS
    return (np.sum(freqs * mag, axis=0, dtype=np.float64) / denom).astype(np.float32)


def extract_article_features(
    y: np.ndarray,
    ref: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    segment_seconds: float,
) -> Dict[str, float]:
    """Возвращает признаки из статьи (без log-Mel)."""
    segment_samples = int(sr * segment_seconds)
    y, ref = _align_pair(y, ref, segment_samples)

    mfcc_y = _mfcc(y, sr, n_mfcc, n_fft, hop_length)
    mfcc_ref = _mfcc(ref, sr, n_mfcc, n_fft, hop_length)

    frame_dist = _framewise_euclidean(mfcc_y, mfcc_ref)
    mfcc_euclidean = float(frame_dist.mean())

    ref_energy = np.sqrt(np.sum(mfcc_ref * mfcc_ref, dtype=np.float64)) + EPS
    mfcc_euclidean_norm = float(mfcc_euclidean / ref_energy)

    vec_y = mfcc_y.mean(axis=1)
    vec_ref = mfcc_ref.mean(axis=1)
    cosine = float(np.dot(vec_y, vec_ref) / ((np.linalg.norm(vec_y) * np.linalg.norm(vec_ref)) + EPS))

    centroid_y = _spectral_centroid(y, sr, n_fft, hop_length)
    centroid_ref = _spectral_centroid(ref, sr, n_fft, hop_length)
    centroid_diff = float(abs(centroid_y.mean() - centroid_ref.mean()))

    noise = y - ref
    noise_energy = float(np.sum(noise * noise, dtype=np.float64))
    signal_energy = float(np.sum(ref * ref, dtype=np.float64)) + EPS
    energy_ratio_noise_to_signal = noise_energy / signal_energy

    return {
        "mfcc_euclidean": mfcc_euclidean,
        "mfcc_euclidean_norm": mfcc_euclidean_norm,
        "mfcc_cosine": cosine,
        "spectral_centroid_diff": centroid_diff,
        "energy_ratio_noise_to_signal": float(energy_ratio_noise_to_signal),
    }
