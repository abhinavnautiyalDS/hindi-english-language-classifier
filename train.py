from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.language_classifier import LABELS, build_linear_svm, build_logistic_regression, load_dataset

ROOT = Path(__file__).resolve().parent


def benchmark_latency(model, texts, repeats=20):
    if not texts:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    model.predict([texts[0]])  # warm-up
    timings = []
    for text in texts:
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict([text])
            timings.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "sample.csv"))
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    args = parser.parse_args()

    data = load_dataset(args.data)
    if data["label"].value_counts().min() < 2:
        raise SystemExit("Need at least two examples per label for stratified split.")
    train_df, test_df = train_test_split(
        data, test_size=args.test_size, random_state=42, stratify=data["label"]
    )
    candidates = {
        "logistic_regression": build_logistic_regression(),
        "linear_svm": build_linear_svm(),
    }
    results, best = {}, None
    for name, model in candidates.items():
        started = time.perf_counter()
        model.fit(train_df["text"], train_df["label"])
        train_seconds = time.perf_counter() - started
        pred = model.predict(test_df["text"])
        macro_f1 = f1_score(test_df["label"], pred, average="macro", zero_division=0)
        latency = benchmark_latency(model, test_df["text"].tolist())
        results[name] = {
            "train_seconds": train_seconds,
            "accuracy": float(accuracy_score(test_df["label"], pred)),
            "macro_f1": float(macro_f1),
            "classification_report": classification_report(
                test_df["label"], pred, labels=list(LABELS), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(
                test_df["label"], pred, labels=list(LABELS)).tolist(),
            "latency": latency,
        }
        print(f"\\n{name}: accuracy={results[name]['accuracy']:.3f}, macro_f1={macro_f1:.3f}, "
              f"p50={latency['p50_ms']:.3f}ms, p95={latency['p95_ms']:.3f}ms")
        if best is None or macro_f1 > best[0]:
            best = (macro_f1, name, model)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "language_classifier.joblib"
    joblib.dump(best[2], model_path)
    (model_dir / "evaluation.json").write_text(json.dumps({
        "warning": "Bundled data is a tiny smoke test; metrics are not production estimates.",
        "selected_model": best[1],
        "labels": list(LABELS),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\\nSelected by test macro-F1: {best[1]}")
    print(f"Saved model: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print("WARNING: replace the tiny sample dataset before trusting any metrics.")


if __name__ == "__main__":
    main()
