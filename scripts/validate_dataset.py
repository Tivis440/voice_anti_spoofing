"""Проверка качества датасета и сплитов для real/fake_engine классификации.

Usage:
  python3 scripts/validate_dataset.py \
    --root . \
    --train data/splits/train.csv \
    --val data/splits/val.csv \
    --test data/splits/test.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf

REQUIRED_COLUMNS = ["path", "class_name", "speaker_id", "utt_id"]


@dataclass
class SplitReport:
    name: str
    df: pd.DataFrame


class ValidationError(Exception):
    pass


def _load_split(root: Path, path: str, name: str) -> SplitReport:
    csv_path = (root / path).resolve()
    if not csv_path.exists():
        raise ValidationError(f"{name}: CSV не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"{name}: нет колонок {missing} в {csv_path}")
    if df.empty:
        raise ValidationError(f"{name}: пустой CSV {csv_path}")
    return SplitReport(name=name, df=df.copy())


def _resolve_paths(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_path"] = out["path"].map(lambda p: str((root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p)).resolve()))
    return out


def _check_files_exist(split: SplitReport) -> list[str]:
    errors = []
    missing = split.df.loc[~split.df["abs_path"].map(lambda p: Path(p).exists())]
    for p in missing["abs_path"].head(20):
        errors.append(f"{split.name}: отсутствует файл {p}")
    if len(missing) > 20:
        errors.append(f"{split.name}: и еще {len(missing) - 20} отсутствующих файлов")
    return errors


def _check_duplicate_rows(split: SplitReport) -> list[str]:
    errors = []
    dup = split.df.duplicated(subset=["abs_path"], keep=False)
    if dup.any():
        errors.append(f"{split.name}: есть дубликаты path ({dup.sum()} строк)")
    return errors


def _check_cross_overlap(reports: Iterable[SplitReport]) -> list[str]:
    reports = list(reports)
    errors = []

    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            a = reports[i]
            b = reports[j]

            a_speakers = set(a.df["speaker_id"].astype(str))
            b_speakers = set(b.df["speaker_id"].astype(str))
            inter_speakers = a_speakers & b_speakers
            if inter_speakers:
                errors.append(
                    f"{a.name} ∩ {b.name}: пересечение speaker_id = {len(inter_speakers)} (утечка)"
                )

            a_pairs = set((a.df["speaker_id"].astype(str) + "::" + a.df["utt_id"].astype(str)).tolist())
            b_pairs = set((b.df["speaker_id"].astype(str) + "::" + b.df["utt_id"].astype(str)).tolist())
            inter_pairs = a_pairs & b_pairs
            if inter_pairs:
                errors.append(f"{a.name} ∩ {b.name}: пересечение (speaker_id, utt_id) = {len(inter_pairs)}")

            a_paths = set(a.df["abs_path"])
            b_paths = set(b.df["abs_path"])
            inter_paths = a_paths & b_paths
            if inter_paths:
                errors.append(f"{a.name} ∩ {b.name}: пересечение path = {len(inter_paths)}")

    return errors


def _class_stats(split: SplitReport) -> pd.Series:
    return split.df["class_name"].value_counts().sort_index()


def _check_balance(split: SplitReport, tolerance: float) -> list[str]:
    errors = []
    counts = _class_stats(split)
    if len(counts) <= 1:
        return [f"{split.name}: найден только 1 класс"]
    min_count = counts.min()
    max_count = counts.max()
    if min_count == 0:
        errors.append(f"{split.name}: есть класс с 0 объектов")
    rel = (max_count - min_count) / max_count
    if rel > tolerance:
        errors.append(
            f"{split.name}: дисбаланс классов {rel:.2%} > {tolerance:.0%} "
            f"(min={min_count}, max={max_count})"
        )
    return errors


def _dur_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    durations = []
    for p in df["abs_path"]:
        try:
            info = sf.info(p)
            durations.append(float(info.duration))
        except Exception:
            continue
    if not durations:
        return 0.0, 0.0, 0.0
    s = pd.Series(durations)
    return float(s.mean()), float(s.median()), float(s.quantile(0.95))


def _print_report(split: SplitReport) -> None:
    print(f"\n[{split.name}] rows={len(split.df)}")
    print("class distribution:")
    counts = _class_stats(split)
    for k, v in counts.items():
        print(f"  {k}: {v}")
    m, med, p95 = _dur_stats(split.df)
    print(f"duration sec: mean={m:.2f} median={med:.2f} p95={p95:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset splits for anti-spoofing")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--train", type=str, default="data/splits/train.csv")
    parser.add_argument("--val", type=str, default="data/splits/val.csv")
    parser.add_argument("--test", type=str, default="data/splits/test.csv")
    parser.add_argument("--imbalance-tolerance", type=float, default=0.10)
    parser.add_argument("--write-template", type=str, default="")
    args = parser.parse_args()

    if args.write_template:
        template = pd.DataFrame(
            [
                {
                    "path": "data/raw/real/spk001/utt0001.wav",
                    "class_name": "real",
                    "speaker_id": "spk001",
                    "utt_id": "utt0001",
                },
                {
                    "path": "data/raw/fake/coqui/spk001/utt0001.wav",
                    "class_name": "fake_coqui",
                    "speaker_id": "spk001",
                    "utt_id": "utt0001",
                },
            ]
        )
        out = Path(args.write_template)
        out.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(out, index=False)
        print(f"Template CSV saved: {out}")
        return 0

    root = Path(args.root).resolve()

    try:
        reports = [
            _load_split(root, args.train, "train"),
            _load_split(root, args.val, "val"),
            _load_split(root, args.test, "test"),
        ]
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        return 2

    for i, rep in enumerate(reports):
        rep.df = _resolve_paths(root, rep.df)
        reports[i] = rep

    errors: list[str] = []
    for rep in reports:
        errors.extend(_check_files_exist(rep))
        errors.extend(_check_duplicate_rows(rep))
        errors.extend(_check_balance(rep, args.imbalance_tolerance))

    errors.extend(_check_cross_overlap(reports))

    print("Dataset validation report")
    print("=" * 32)
    for rep in reports:
        _print_report(rep)

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("\nOK: dataset and splits passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
