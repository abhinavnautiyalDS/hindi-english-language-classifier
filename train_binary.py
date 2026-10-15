from __future__ import annotations
import json
from pathlib import Path
import joblib
from src.binary_training import binary_corpus, evaluate_candidates, with_other_rejection

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data/processed/train.csv"
TEST_PATH = ROOT / "data/processed/test.csv"
MODEL_PATH = ROOT / "models/english_hinglish_classifier.joblib"
REPORT_PATH = ROOT / "models/binary_evaluation.json"


def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise SystemExit("Processed datasets are missing; run scripts/build_dataset.py first.")
    train = binary_corpus(TRAIN_PATH)
    test = binary_corpus(TEST_PATH)
    test_texts = set(test["text"].str.casefold())
    train = train[~train["text"].str.casefold().isin(test_texts)].drop_duplicates("text").reset_index(drop=True)
    print(f"Binary training rows: {len(train):,}")
    print(f"Binary held-out corpus rows: {len(test):,}")
    print("Train labels:", train["label"].value_counts().to_dict())
    print("Test labels:", test["label"].value_counts().to_dict())
    results, models = evaluate_candidates(train, test)
    ranking = sorted(results, key=lambda name: (-results[name]["macro_f1"], results[name]["latency"]["p95_ms"], results[name]["model_size_mb"]))
    selected = ranking[0]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Persist the selected binary model behind a reject layer so unsupported scripts
    # and low-margin inputs return OTHER instead of being forced into a binary class.
    joblib.dump(with_other_rejection(models[selected], min_margin=0.35), MODEL_PATH)
    report = {
        "task": "English vs Hinglish with OTHER rejection",
        "labels": ["english", "hinglish", "other"],
        "training_labels": ["english", "hinglish"],
        "other_policy": "Non-Latin-script letters -> other; Latin-script LinearSVC margin < 0.35 -> other. This is not a universal language detector.",
        "dataset": "Repository processed corpus filtered to English and Hinglish; pure Hindi excluded",
        "evaluation_note": "Binary corpus metrics evaluate English/Hinglish only. OTHER rejection needs separate multilingual, manually labelled validation before production claims.",
        "selected_model": selected,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_label_counts": train["label"].value_counts().to_dict(),
        "test_label_counts": test["label"].value_counts().to_dict(),
        "ranking": ranking,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Selected:", selected)
    for name in ranking:
        row = results[name]
        print(f"{name}: accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} p95={row['latency']['p95_ms']:.3f}ms size={row['model_size_mb']:.3f}MB")
    print("OTHER checks:")
    selected_model = joblib.load(MODEL_PATH)
    for example in ["నమస్తే అండి, నేను SRC University నుంచి", "नमस्ते", "Hello, how are you?", "Sir after 1 hour", "જી going"]:
        print(repr(example), "->", selected_model.predict([example])[0])
    print(f"Saved {MODEL_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
