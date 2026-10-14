from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

LABELS = ("hindi", "english", "hinglish")
DEFAULT_MODEL_PATH = Path("models/language_classifier.joblib")


def normalize_text(text: str) -> str:
    """Conservative normalization: preserve script and spelling cues."""
    return re.sub(r"\\s+", " ", str(text or "").strip().lower())


def build_logistic_regression() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True, max_features=100_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                                 sublinear_tf=True, max_features=150_000)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=42)),
    ])


def build_linear_svm() -> Pipeline:
    from sklearn.svm import LinearSVC

    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True, max_features=100_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                                 sublinear_tf=True, max_features=150_000)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
    ])


def load_dataset(path: str | Path):
    import pandas as pd

    data = pd.read_csv(path)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("CSV must contain columns named 'text' and 'label'.")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].map(normalize_text)
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["text"].str.len() > 0]
    invalid = set(data["label"]) - set(LABELS)
    if invalid:
        raise ValueError(f"Unexpected labels: {sorted(invalid)}. Allowed: {LABELS}")
    if data["label"].nunique() != len(LABELS):
        raise ValueError(f"Dataset must contain all three labels: {LABELS}")
    return data.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    uncertain: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LanguageDetector:
    def __init__(self, model: Any, min_confidence: float = 0.55, min_chars: int = 4):
        self.model = model
        self.min_confidence = float(min_confidence)
        self.min_chars = int(min_chars)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH, **kwargs: Any):
        return cls(joblib.load(path), **kwargs)

    def predict(self, text: str) -> Prediction:
        cleaned = normalize_text(text)
        start = time.perf_counter()
        if len(cleaned) < self.min_chars:
            return Prediction("uncertain", 0.0, {},
                              (time.perf_counter() - start) * 1000, True)
        probabilities = self.model.predict_proba([cleaned])[0]
        classes = list(self.model.classes_)
        idx = int(probabilities.argmax())
        confidence = float(probabilities[idx])
        elapsed = (time.perf_counter() - start) * 1000
        uncertain = confidence < self.min_confidence
        return Prediction(
            "uncertain" if uncertain else str(classes[idx]),
            confidence,
            {str(k): float(v) for k, v in zip(classes, probabilities)},
            elapsed,
            uncertain,
        )


def save_model(model: Any, path: str | Path = DEFAULT_MODEL_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
