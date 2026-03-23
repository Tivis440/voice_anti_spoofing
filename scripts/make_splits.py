from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build train/val/test splits from data/raw or data")
    p.add_argument("--root", default=".")
    p.add_argument("--real-dir", default="data/raw/real")
    p.add_argument("--fake-dir", default="data/raw/fake")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--exts", default="wav,flac,mp3,opus,m4a")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=0)
    return p.parse_args()


def collect_rows(root: Path, real_dir: Path, fake_dir: Path, exts: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    if real_dir.exists():
        for p in real_dir.rglob("*"):
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                rows.append(
                    {
                        "path": str(p.relative_to(root)),
                        "class_name": "real",
                        "speaker_id": p.parent.name,
                        "utt_id": p.stem,
                    }
                )

    if fake_dir.exists():
        for eng_dir in fake_dir.iterdir():
            if not eng_dir.is_dir():
                continue
            cls = f"fake_{eng_dir.name}"
            for p in eng_dir.rglob("*"):
                if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                    rows.append(
                        {
                            "path": str(p.relative_to(root)),
                            "class_name": cls,
                            "speaker_id": p.parent.name,
                            "utt_id": p.stem,
                        }
                    )
    return rows


def maybe_limit_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    if max_per_class <= 0:
        return df
    parts = []
    for _, g in df.groupby("class_name", sort=True):
        parts.append(g if len(g) <= max_per_class else g.sample(n=max_per_class, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def split_by_speaker(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    speakers = df["speaker_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    sp_train = set(speakers[:n_train])
    sp_val = set(speakers[n_train:n_train + n_val])
    sp_test = set(speakers[n_train + n_val:])

    train = df[df["speaker_id"].astype(str).isin(sp_train)].copy()
    val = df[df["speaker_id"].astype(str).isin(sp_val)].copy()
    test = df[df["speaker_id"].astype(str).isin(sp_test)].copy()
    return train, val, test


def split_stratified_rows(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for _, g in df.groupby("class_name", sort=True):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 0:
                n_val -= 1

        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train : n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val :])

    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return train, val, test


def resolve_data_dirs(root: Path, real_arg: str, fake_arg: str) -> tuple[Path, Path]:
    real_dir = (root / real_arg).resolve()
    fake_dir = (root / fake_arg).resolve()

    if not real_dir.exists():
        fallback_real = (root / "data" / "real").resolve()
        if fallback_real.exists():
            real_dir = fallback_real

    if not fake_dir.exists():
        fallback_fake = (root / "data" / "fake").resolve()
        if fallback_fake.exists():
            fake_dir = fallback_fake

    return real_dir, fake_dir


def is_valid_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, classes: set[str]) -> bool:
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        return False
    for split in (train, val, test):
        if set(split["class_name"]) != classes:
            return False
    return True


def main() -> int:
    args = parse_args()

    root = Path(args.root).resolve()
    real_dir, fake_dir = resolve_data_dirs(root, args.real_dir, args.fake_dir)
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()}

    rows = collect_rows(root, real_dir, fake_dir, exts)
    if not rows:
        raise ValueError(
            "No files found in real/fake directories. "
            f"Tried real_dir={real_dir}, fake_dir={fake_dir}"
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    df = maybe_limit_per_class(df, args.max_per_class, args.seed)
    all_classes = set(df["class_name"])

    train, val, test = split_by_speaker(df, args.train_ratio, args.val_ratio, args.seed)
    strategy = "speaker"
    if not is_valid_split(train, val, test, all_classes):
        train, val, test = split_stratified_rows(df, args.train_ratio, args.val_ratio, args.seed)
        strategy = "stratified_rows_fallback"

    df.to_csv(out_dir / "all.csv", index=False)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(f"real_dir: {real_dir}")
    print(f"fake_dir: {fake_dir}")
    print(f"split_strategy: {strategy}")
    print("all:", len(df))
    print(df["class_name"].value_counts().sort_index())
    print("train:", len(train), "val:", len(val), "test:", len(test))
    print(f"saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
