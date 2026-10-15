from __future__ import annotations
import io, re, time, unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

LABELS = ("english", "hinglish")
OUTPUT_LABELS = ("english", "hinglish", "other")


def script_profile(text: str) -> tuple[bool, bool]:
    """Return (has_non_latin_letters, has_latin_letters)."""
    has_non_latin = has_latin = False
    for ch in str(text or ""):
        if ch.isalpha():
            if "LATIN" in unicodedata.name(ch, ""):
                has_latin = True
            else:
                has_non_latin = True
    return has_non_latin, has_latin


class OtherLanguageRejector:
    """Binary English/Hinglish model with an explicit OTHER/reject outcome.

    Non-Latin-script utterances are OTHER. For Latin-script text, low-margin
    predictions are OTHER rather than forcing a potentially unrelated language
    into English or Hinglish. This is a reject option, not universal language ID.
    """
    def __init__(self, model, min_margin: float = 0.35):
        self.model = model
        self.min_margin = float(min_margin)
        self.classes_ = np.array(OUTPUT_LABELS, dtype=object)

    def predict(self, texts):
        outputs = []
        for text in texts:
            cleaned = str(text or "").strip()
            has_non_latin, has_latin = script_profile(cleaned)
            if not cleaned or (has_non_latin and not has_latin):
                outputs.append("other")
                continue
            if has_non_latin and has_latin:
                outputs.append("hinglish")
                continue
            score = float(np.asarray(self.model.decision_function([cleaned])).reshape(-1)[0])
            if abs(score) < self.min_margin:
                outputs.append("other")
            else:
                # LinearSVC decision score is positive for classes_[1].
                outputs.append(str(self.model.classes_[1] if score > 0 else self.model.classes_[0]))
        return np.asarray(outputs, dtype=object)

    def predict_proba(self, texts):
        # Confidence-like scores for UI display; OTHER is a reject score, not calibrated probability.
        rows = []
        for text in texts:
            cleaned = str(text or "").strip()
            has_non_latin, has_latin = script_profile(cleaned)
            if not cleaned or (has_non_latin and not has_latin):
                rows.append([0.0, 0.0, 1.0])
                continue
            if has_non_latin and has_latin:
                rows.append([0.0, 1.0, 0.0])
                continue
            score = float(np.asarray(self.model.decision_function([cleaned])).reshape(-1)[0])
            strength = 1.0 / (1.0 + np.exp(-min(abs(score), 30.0)))
            if abs(score) < self.min_margin:
                rows.append([0.0, 0.0, 1.0])
            elif score > 0:
                rows.append([1.0-strength, strength, 0.0])
            else:
                rows.append([strength, 1.0-strength, 0.0])
        return np.asarray(rows, dtype=float)


def with_other_rejection(model, min_margin: float = 0.35):
    return OtherLanguageRejector(model, min_margin=min_margin)


def normalize_binary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("CSV must contain text and label columns.")
    data = frame.copy()
    data["text"] = data["text"].fillna("").astype(str).str.lower().str.replace(r"\\s+", " ", regex=True).str.strip()
    data["label"] = data["label"].fillna("").astype(str).str.strip().str.lower()
    aliases = {"eng": "english", "en": "english", "mixed": "hinglish"}
    data["label"] = data["label"].map(lambda x: aliases.get(x, x))
    data = data[(data["text"].str.len() >= 2) & data["label"].isin(LABELS)].copy()
    if data["label"].nunique() != 2:
        raise ValueError("Need verified examples for both labels: english and hinglish.")
    return data.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


def binary_corpus(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain text,label columns.")
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    return normalize_binary_frame(frame[frame["label"].isin(LABELS)].copy())


def make_candidates():
    def features():
        return FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=150_000, sublinear_tf=True)),
        ])
    return {
        "logistic_regression": Pipeline([("features", features()), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0, random_state=42))]),
        "linear_svm": Pipeline([("features", features()), ("classifier", LinearSVC(class_weight="balanced", C=1.0, random_state=42))]),
        "sgd_log_loss": Pipeline([("features", features()), ("classifier", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1500, class_weight="balanced", random_state=42, tol=1e-3))]),
        "complement_nb": Pipeline([("features", features()), ("classifier", ComplementNB(alpha=0.5))]),
    }


def benchmark_latency(model, texts, repeats=3, max_examples=500):
    samples = list(texts[:max_examples])
    if not samples:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "n_calls": 0}
    model.predict([samples[0]])
    timings = []
    for _ in range(repeats):
        for text in samples:
            start = time.perf_counter()
            model.predict([text])
            timings.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)), "n_calls": len(timings)}


def evaluate_candidates(train: pd.DataFrame, test: pd.DataFrame):
    if set(train["label"]) != set(LABELS) or set(test["label"]) != set(LABELS):
        raise ValueError("Both train and test must contain English and Hinglish.")
    overlap = set(train["text"].str.casefold()) & set(test["text"].str.casefold())
    if overlap:
        raise ValueError(f"Leakage check failed: {len(overlap)} exact texts overlap train/test.")
    results, fitted = {}, {}
    for name, model in make_candidates().items():
        start = time.perf_counter()
        model.fit(train["text"], train["label"])
        train_seconds = time.perf_counter() - start
        train_pred = model.predict(train["text"])
        test_pred = model.predict(test["text"])
        buf = io.BytesIO()
        joblib.dump(model, buf)
        size_mb = len(buf.getvalue()) / (1024 * 1024)
        train_acc = accuracy_score(train["label"], train_pred)
        test_acc = accuracy_score(test["label"], test_pred)
        results[name] = {
            "accuracy": float(test_acc),
            "macro_f1": float(f1_score(test["label"], test_pred, labels=list(LABELS), average="macro", zero_division=0)),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(f1_score(train["label"], train_pred, labels=list(LABELS), average="macro", zero_division=0)),
            "generalization_gap_accuracy": float(train_acc - test_acc),
            "train_seconds": train_seconds,
            "model_size_mb": size_mb,
            "latency": benchmark_latency(model, test["text"].tolist()),
            "classification_report": classification_report(test["label"], test_pred, labels=list(LABELS), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(test["label"], test_pred, labels=list(LABELS)).tolist(),
            "test_rows": len(test),
        }
        fitted[name] = model
    return results, fitted
