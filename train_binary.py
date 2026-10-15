from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from src.binary_training import binary_corpus, evaluate_candidates

ROOT=Path(__file__).resolve().parent
TRAIN_PATH=ROOT/"data/processed/train.csv"
TEST_PATH=ROOT/"data/processed/test.csv"
MODEL_PATH=ROOT/"models/english_hinglish_classifier.joblib"
REPORT_PATH=ROOT/"models/binary_evaluation.json"

def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise SystemExit("Processed datasets are missing; run scripts/build_dataset.py first.")
    train=binary_corpus(TRAIN_PATH)
    test=binary_corpus(TEST_PATH)
    overlap=set(train["text"].str.casefold()) & set(test["text"].str.casefold())
    if overlap:
        train=train[~train["text"].str.casefold().isin(set(test["text"].str.casefold()))].reset_index(drop=True)
    print(f"Binary training rows: {len(train):,}")
    print(f"Binary held-out corpus rows: {len(test):,}")
    print("Train labels:",train["label"].value_counts().to_dict())
    print("Test labels:",test["label"].value_counts().to_dict())
    results,models=evaluate_candidates(train,test)
    ranking=sorted(results,key=lambda name:(-results[name]["macro_f1"],results[name]["latency"]["p95_ms"],results[name]["model_size_mb"]))
    selected=ranking[0]
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(models[selected],MODEL_PATH)
    report={
        "task":"binary English vs Hinglish",
        "labels":["english","hinglish"],
        "dataset":"Repository processed corpus filtered to English and Hinglish; pure Hindi excluded",
        "evaluation_note":"Public-corpus held-out scores are not production-call validation. Validate on manually labeled, speaker/call-separated STT transcripts.",
        "selected_model":selected,
        "train_rows":len(train),
        "test_rows":len(test),
        "train_label_counts":train["label"].value_counts().to_dict(),
        "test_label_counts":test["label"].value_counts().to_dict(),
        "ranking":ranking,
        "results":results,
    }
    REPORT_PATH.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Selected:",selected)
    for name in ranking:
        row=results[name]
        print(f"{name}: accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
              f"p95={row['latency']['p95_ms']:.3f}ms size={row['model_size_mb']:.3f}MB "
              f"train-test-gap={row['generalization_gap_accuracy']:.4f}")
    print(f"Saved {MODEL_PATH} and {REPORT_PATH}")

if __name__=="__main__":
    main()
