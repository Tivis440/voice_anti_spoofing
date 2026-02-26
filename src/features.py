"""Однофайловые признаки в стиле статьи (без log-Mel)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Dict

import audioread
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import lfilter, resample_poly, stft

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
    """Загружает mono-аудио и приводит к `sample_rate`."""
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


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmin, fmax = 0.0, sr / 2.0
    mels = np.linspace(
        _hz_to_mel(np.array([fmin]))[0],
        _hz_to_mel(np.array([fmax]))[0],
        n_mels + 2,
    )
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


def _mfcc_frames(y: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    _, _, zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, padded=False)
    power = (np.abs(zxx) ** 2).astype(np.float32)
    mel_fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=max(26, n_mfcc * 2))
    mel_spec = np.dot(mel_fb, power)
    log_mel = np.log(mel_spec + EPS)
    coeffs = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return coeffs.astype(np.float32)


def _delta(frames: np.ndarray) -> np.ndarray:
    if frames.shape[1] < 2:
        return np.zeros_like(frames)
    kernel = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    return lfilter(kernel, [1.0], frames, axis=1).astype(np.float32)


def _spectral_stats(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> dict[str, np.ndarray]:
    freqs, _, zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, padded=False)
    mag = np.abs(zxx).astype(np.float32)
    power = mag * mag
    sum_mag = np.sum(mag, axis=0, dtype=np.float64) + EPS
    sum_power = np.sum(power, axis=0, dtype=np.float64) + EPS

    centroid = np.sum(freqs[:, None] * mag, axis=0, dtype=np.float64) / sum_mag
    bandwidth = np.sqrt(
        np.sum(((freqs[:, None] - centroid[None, :]) ** 2) * mag, axis=0, dtype=np.float64) / sum_mag
    )

    cumsum = np.cumsum(power, axis=0)
    rolloff_level = 0.85 * sum_power
    rolloff_idx = np.argmax(cumsum >= rolloff_level[None, :], axis=0)
    rolloff = freqs[rolloff_idx]

    flatness = np.exp(np.mean(np.log(mag + EPS), axis=0)) / (np.mean(mag, axis=0) + EPS)

    return {
        "centroid": centroid.astype(np.float32),
        "bandwidth": bandwidth.astype(np.float32),
        "rolloff": rolloff.astype(np.float32),
        "flatness": flatness.astype(np.float32),
    }


def _zcr(y: np.ndarray, frame_len: int, hop_length: int) -> np.ndarray:
    if len(y) < frame_len:
        y = _center_crop_or_pad(y, frame_len)
    out = []
    for start in range(0, len(y) - frame_len + 1, hop_length):
        frame = y[start : start + frame_len]
        signs = np.signbit(frame)
        crossings = np.sum(signs[:-1] != signs[1:])
        out.append(crossings / max(1, frame_len - 1))
    if not out:
        out = [0.0]
    return np.asarray(out, dtype=np.float32)


def _rms(y: np.ndarray, frame_len: int, hop_length: int) -> np.ndarray:
    if len(y) < frame_len:
        y = _center_crop_or_pad(y, frame_len)
    out = []
    for start in range(0, len(y) - frame_len + 1, hop_length):
        frame = y[start : start + frame_len]
        out.append(np.sqrt(np.mean(frame * frame) + EPS))
    if not out:
        out = [0.0]
    return np.asarray(out, dtype=np.float32)


def _summary_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
    }


def extract_article_features_single(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    segment_seconds: float,
) -> Dict[str, float]:
    """Single-file признаки на основе групп из статьи."""
    segment_samples = int(sr * segment_seconds)
    y = _center_crop_or_pad(y, segment_samples)

    mfcc = _mfcc_frames(y, sr, n_mfcc, n_fft, hop_length)
    d1 = _delta(mfcc)
    d2 = _delta(d1)

    features: Dict[str, float] = {}
    for i in range(n_mfcc):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_d1_{i}_mean"] = float(np.mean(d1[i]))
        features[f"mfcc_d2_{i}_mean"] = float(np.mean(d2[i]))

    spec = _spectral_stats(y, sr, n_fft, hop_length)
    features.update(_summary_stats(spec["centroid"], "centroid"))
    features.update(_summary_stats(spec["bandwidth"], "bandwidth"))
    features.update(_summary_stats(spec["rolloff"], "rolloff"))
    features.update(_summary_stats(spec["flatness"], "flatness"))

    frame_len = n_fft
    zcr = _zcr(y, frame_len, hop_length)
    rms = _rms(y, frame_len, hop_length)
    features.update(_summary_stats(zcr, "zcr"))
    features.update(_summary_stats(rms, "rms"))

    signal_energy = float(np.sum(y * y, dtype=np.float64)) + EPS
    spectral_energy = float(np.mean(spec["bandwidth"]))
    features["energy_ratio_proxy"] = float(spectral_energy / signal_energy)

    return features
