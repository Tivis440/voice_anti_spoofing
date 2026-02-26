"""Классическая модель (по признакам статьи) для binary real/fake."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    val_accuracy: float
    report: str


class ArticleBinaryClassifier:
    def __init__(self, random_state: int = 42, max_iter: int = 2000):
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        random_state=random_state,
                        max_iter=max_iter,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.pipeline.fit(x_train, y_train)

    def evaluate(self, x_val: np.ndarray, y_val: np.ndarray) -> TrainResult:
        pred = self.pipeline.predict(x_val)
        acc = float(accuracy_score(y_val, pred))
        report = classification_report(y_val, pred, digits=4)
        return TrainResult(val_accuracy=acc, report=report)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(x)

    def save(self, path: str | Path, payload: Dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str | Path) -> Dict:
        return joblib.load(path)
