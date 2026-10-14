from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle

from src.language_classifier import LABELS, build_linear_svm, build_logistic_regression, normalize_text

ROOT = Path(__file__).resolve().parent


def read_labeled_csv(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError(f"{path} must have 'text' and 'label' columns.")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].map(normalize_text)
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[(data["text"].str.len() >= 2) & data["label"].isin(LABELS)]
    if data.empty:
        raise ValueError(f"No valid labeled rows in {path}")
    return data.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


def benchmark_latency(model, texts, repeats=5, max_examples=500):
    sample = list(texts[:max_examples])
    if not sample:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "n_calls": 0}
    model.predict([sample[0]])
    timings = []
    for _ in range(repeats):
        for text in sample:
            start = time.perf_counter()
            model.predict([text])
            timings.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)), "n_calls": len(timings)}


def score_model(model, train_df, test_df):
    started = time.perf_counter()
    model.fit(train_df["text"], train_df["label"])
    train_seconds = time.perf_counter() - started
    pred = model.predict(test_df["text"])
    latency = benchmark_latency(model, test_df["text"].tolist())
    return model, {
        "train_seconds": train_seconds,
        "accuracy": float(accuracy_score(test_df["label"], pred)),
        "macro_f1": float(f1_score(test_df["label"], pred, labels=list(LABELS),
                                    average="macro", zero_division=0)),
        "classification_report": classification_report(
            test_df["label"], pred, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test_df["label"], pred, labels=list(LABELS)).tolist(),
        "latency": latency,
        "test_rows": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=str(ROOT / "data" / "processed" / "train.csv"))
    parser.add_argument("--test-data", default=str(ROOT / "data" / "processed" / "test.csv"))
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Optional debugging cap; 0 means use all cleaned training data.")
    args = parser.parse_args()

    train_df = read_labeled_csv(args.train_data)
    test_df = read_labeled_csv(args.test_data)
    overlap = set(train_df["text"].str.casefold()) & set(test_df["text"].str.casefold())
    if overlap:
        raise SystemExit(f"Leakage check failed: {len(overlap)} exact texts overlap train/test.")
    if set(train_df["label"]) != set(LABELS):
        raise SystemExit(f"Training split must contain all labels {LABELS}; got {train_df['label'].value_counts().to_dict()}")
    if set(test_df["label"]) != set(LABELS):
        raise SystemExit(f"Test split must contain all labels {LABELS}; got {test_df['label'].value_counts().to_dict()}")
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        per_class = max(1, args.max_train_rows // len(LABELS))
        train_df = shuffle(train_df, random_state=42).groupby("label", group_keys=False).head(per_class).reset_index(drop=True)

    print(f"Clean training rows: {len(train_df):,}")
    print(f"Official held-out test rows: {len(test_df):,}")
    print("Train class counts:", train_df["label"].value_counts().to_dict())
    print("Test class counts:", test_df["label"].value_counts().to_dict())

    results, fitted = {}, {}
    for name, model in {
        "logistic_regression": build_logistic_regression(),
        "linear_svm": build_linear_svm(),
    }.items():
        fitted[name], results[name] = score_model(model, train_df, test_df)
        print(f"{name}: accuracy={results[name]['accuracy']:.4f}, "
              f"macro_f1={results[name]['macro_f1']:.4f}, "
              f"p50={results[name]['latency']['p50_ms']:.3f}ms, "
              f"p95={results[name]['latency']['p95_ms']:.3f}ms")

    # Serve Logistic Regression because its confidence values are available to the app.
    deploy_name = "logistic_regression"
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "language_classifier.joblib"
    joblib.dump(fitted[deploy_name], model_path)
    audit_path = ROOT / "data" / "processed" / "dataset_audit.json"
    dataset_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    evaluation = {
        "dataset": dataset_audit.get("dataset", "unknown"),
        "source_url": dataset_audit.get("source_url"),
        "sources": dataset_audit.get("sources", []),
        "license": dataset_audit.get("license"),
        "evaluation_split": "COMI-LINGUA held-out split plus a paired-row PHINC holdout; exact train/test text overlap removed",
        "selected_deployment_model": deploy_name,
        "best_model_by_test_macro_f1": max(results, key=lambda key: results[key]["macro_f1"]),
        "labels": list(LABELS),
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "train_class_counts": train_df["label"].value_counts().to_dict(),
        "test_class_counts": test_df["label"].value_counts().to_dict(),
        "results": results,
        "important_limitations": [
            "Metrics describe the source text corpus, not the target production STT engine.",
            "Spoken-call transcripts and STT errors require a separate held-out evaluation set.",
            "Classifier latency excludes STT, network, LLM, and telephony latency.",
        ],
    }
    (model_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved deployable model: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print("Evaluation saved to models/evaluation.json")


if __name__ == "__main__":
    main()
