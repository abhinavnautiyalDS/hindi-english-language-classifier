from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

LABELS = ("english", "hinglish")


def normalize_binary_frame(frame: pd.DataFrame, *, require_labels: bool = True) -> pd.DataFrame:
    if "text" not in frame.columns:
        raise ValueError("CSV must contain a 'text' column.")
    data = frame.copy()
    data["text"] = data["text"].fillna("").astype(str).str.lower().str.replace(r"\\s+", " ", regex=True).str.strip()
    data = data[data["text"].str.len() >= 2].copy()
    if require_labels:
        if "label" not in data.columns:
            raise ValueError("Labeled training CSV must contain 'text' and 'label' columns.")
        data["label"] = data["label"].fillna("").astype(str).str.strip().str.lower()
        aliases = {"eng": "english", "en": "english", "hinglish": "hinglish", "mixed": "hinglish"}
        data["label"] = data["label"].map(lambda x: aliases.get(x, x))
        data = data[data["label"].isin(LABELS)].copy()
        if data["label"].nunique() != 2:
            raise ValueError("Need verified examples for both labels: english and hinglish.")
    return data.drop_duplicates(subset=["text"] + (["label"] if "label" in data.columns else [])).reset_index(drop=True)


def binary_corpus(path: str | Path) -> pd.DataFrame:
    """Load the repository's existing labeled corpus and retain only target classes."""
    frame = pd.read_csv(path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain text,label columns.")
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame = frame[frame["label"].isin(LABELS)].copy()  # Pure Hindi is excluded, not relabeled as Hinglish.
    return normalize_binary_frame(frame)


def make_candidates():
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=150_000, sublinear_tf=True)),
    ])
    # Fresh independent feature pipelines per estimator, avoiding fitted-state sharing.
    def make_features():
        return FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=150_000, sublinear_tf=True)),
        ])
    return {
        "logistic_regression": Pipeline([("features", make_features()), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0, random_state=42))]),
        "linear_svm": Pipeline([("features", make_features()), ("classifier", LinearSVC(class_weight="balanced", C=1.0, random_state=42))]),
        "sgd_log_loss": Pipeline([("features", make_features()), ("classifier", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1500, class_weight="balanced", random_state=42, tol=1e-3))]),
        "complement_nb": Pipeline([("features", make_features()), ("classifier", ComplementNB(alpha=0.5))]),
    }


def benchmark_latency(model, texts, repeats: int = 3, max_examples: int = 500):
    samples = list(texts[:max_examples])
    if not samples:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "n_calls": 0}
    model.predict([samples[0]])
    times = []
    for _ in range(repeats):
        for text in samples:
            start = time.perf_counter()
            model.predict([text])
            times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)), "n_calls": len(times)}


def evaluate_candidates(train: pd.DataFrame, test: pd.DataFrame):
    if set(train["label"]) != set(LABELS) or set(test["label"]) != set(LABELS):
        raise ValueError("Both train and test must include English and Hinglish.")
    overlap = set(train["text"].str.casefold()) & set(test["text"].str.casefold())
    if overlap:
        raise ValueError(f"Exact text leakage detected: {len(overlap)} overlapping examples.")
    results, fitted = {}, {}
    for name, model in make_candidates().items():
        start = time.perf_counter()
        model.fit(train["text"], train["label"])
        train_seconds = time.perf_counter() - start
        pred = model.predict(test["text"])
        report = classification_report(test["label"], pred, labels=list(LABELS), output_dict=True, zero_division=0)
        results[name] = {
            "accuracy": float(accuracy_score(test["label"], pred)),
            "macro_f1": float(f1_score(test["label"], pred, labels=list(LABELS), average="macro", zero_division=0)),
            "train_seconds": train_seconds,
            "latency": benchmark_latency(model, test["text"].tolist()),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(test["label"], pred, labels=list(LABELS)).tolist(),
            "test_rows": len(test),
        }
        fitted[name] = model
    return results, fitted
