"""
Создаёт тестовые wav-файлы в data/raw/real и data/raw/fake,
чтобы можно было сразу запустить обучение.
После подстановки своих данных — замените пути в data/splits/*.csv.
"""

import math
import struct
import wave
from pathlib import Path

# Корень проекта (скрипт из scripts/)
ROOT = Path(__file__).resolve().parent.parent
SR = 16000
DURATION = 2.5  # чуть длиннее segment_length


def make_tone(freq: float, duration: float, sr: int) -> bytes:
    n = int(sr * duration)
    frames = []
    for i in range(n):
        t = i / sr
        sample = 0.3 * math.sin(2 * math.pi * freq * t)
        frames.append(struct.pack("<h", int(max(-32768, min(32767, sample * 32767)))))
    return b"".join(frames)


def main():
    for kind, subdir in [("real", "real"), ("fake", "fake")]:
        d = ROOT / "data" / "raw" / subdir
        d.mkdir(parents=True, exist_ok=True)
        freq = 440.0 if kind == "real" else 330.0
        for i in range(1, 4):
            with wave.open(str(d / f"sample{i}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(make_tone(freq, DURATION, SR))
    print("Created data/raw/real/sample1-3.wav and data/raw/fake/sample1-3.wav")


if __name__ == "__main__":
    main()
