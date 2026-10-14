import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.language_classifier import LABELS, LanguageDetector, build_logistic_regression, build_linear_svm, load_dataset

ROOT = Path(__file__).resolve().parent
PROCESSED_TRAIN = ROOT / "data" / "processed" / "train.csv"
DATA_PATH = PROCESSED_TRAIN if PROCESSED_TRAIN.exists() else ROOT / "data" / "sample.csv"
MODEL_PATH = ROOT / "models" / "language_classifier.joblib"
EVALUATION_PATH = ROOT / "models" / "evaluation.json"

st.set_page_config(page_title="Hindi-English Language Classifier", page_icon="🗣️", layout="wide")
st.title("Hindi · English · Hinglish Classifier")
st.caption("Low-latency sentence-level language detection for voice-agent STT transcripts.")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Optional labeled CSV path", str(DATA_PATH))
    min_confidence = st.slider("Minimum confidence", 0.30, 0.95, 0.55, 0.05)
    min_chars = st.slider("Minimum characters", 1, 12, 4)

@st.cache_resource
def get_model(model_path, csv_path):
    if Path(model_path).exists():
        import joblib
        return joblib.load(model_path), "Trained corpus model"
    data = load_dataset(csv_path)
    model = build_logistic_regression()
    model.fit(data["text"], data["label"])
    return model, "Fallback demo model (not production-trained)"

try:
    model, model_source = get_model(str(MODEL_PATH), data_path)
    detector = LanguageDetector(model, min_confidence=min_confidence, min_chars=min_chars)
except Exception as exc:
    st.error(f"Unable to load model or dataset: {exc}")
    st.stop()

st.caption(f"Model source: {model_source}")
single, batch, evaluate, integration = st.tabs(["Single transcript", "Batch CSV", "Held-out evaluation", "Integration"])

with single:
    transcript = st.text_area("STT transcript", placeholder="Sir mujhe fees ke baare mein details chahiye")
    if st.button("Detect language", type="primary") and transcript.strip():
        result = detector.predict(transcript)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", result.label.upper())
        c2.metric("Confidence", f"{result.confidence:.1%}")
        c3.metric("Classifier latency", f"{result.latency_ms:.3f} ms")
        if result.probabilities:
            st.bar_chart(pd.Series(result.probabilities, name="Probability"))
        if result.uncertain:
            st.warning("Ambiguous or short input. Consider preserving the previous conversation language.")
        st.json(result.to_dict())

with batch:
    uploaded = st.file_uploader("Upload CSV with a text column", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df:
            st.error("CSV must contain a text column.")
        else:
            predictions = [detector.predict(str(t)) for t in df["text"].fillna("")]
            df["predicted_label"] = [p.label for p in predictions]
            df["confidence"] = [p.confidence for p in predictions]
            df["latency_ms"] = [p.latency_ms for p in predictions]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download predictions", df.to_csv(index=False).encode("utf-8"), "language_predictions.csv", "text/csv")
            if "label" in df:
                valid = df["label"].astype(str).str.lower().isin(LABELS)
                if valid.any():
                    st.metric("Accuracy", f"{accuracy_score(df.loc[valid, "label"].str.lower(), df.loc[valid, "predicted_label"]):.1%}")

with evaluate:
    if EVALUATION_PATH.exists():
        evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
        st.write(f"Dataset: {evaluation.get("dataset", "Unknown")}")
        st.write(f"Official held-out test rows: {evaluation.get("test_rows", "Unknown"):,}")
        st.caption("These corpus metrics are not a substitute for evaluation on consented transcripts from your actual STT engine.")
        for name, result in evaluation.get("results", {}).items():
            st.markdown(f"### {name.replace("_", " ").title()}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{result.get("accuracy", 0):.2%}")
            c2.metric("Macro-F1", f"{result.get("macro_f1", 0):.3f}")
            c3.metric("p50 latency", f"{result.get("latency", {}).get("p50_ms", 0):.3f} ms")
            c4.metric("p95 latency", f"{result.get("latency", {}).get("p95_ms", 0):.3f} ms")
            report = result.get("classification_report", {})
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            matrix = result.get("confusion_matrix")
            if matrix:
                st.write("Confusion matrix (rows = true; columns = predicted)")
                st.dataframe(pd.DataFrame(matrix, index=LABELS, columns=LABELS))
        st.caption(f"Deployment model: {evaluation.get("selected_deployment_model", "unknown")}")
    else:
        st.warning("No corpus evaluation report found. Run python scripts/build_dataset.py and python train.py before interpreting model quality.")

    with st.expander("Run a separate quick comparison on a supplied labeled CSV"):
        st.warning("This uses a random holdout from the selected CSV; it is not the official test result. Do not use the tiny sample file to claim accuracy.")
        if st.button("Run local CSV comparison"):
            try:
                data = load_dataset(data_path)
                if data["label"].value_counts().min() < 2:
                    st.error("Need at least two examples per class.")
                else:
                    tr, te = train_test_split(data, test_size=0.30, random_state=42, stratify=data["label"])
                    summaries = []
                    for name, builder in [("Logistic Regression", build_logistic_regression), ("Linear SVM", build_linear_svm)]:
                        candidate = builder()
                        candidate.fit(tr["text"], tr["label"])
                        pred = candidate.predict(te["text"])
                        summaries.append({"model": name, "accuracy": accuracy_score(te["label"], pred), "macro_f1": f1_score(te["label"], pred, average="macro", zero_division=0)})
                    st.dataframe(pd.DataFrame(summaries), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

with integration:
    st.code("""from src.language_classifier import LanguageDetector

detector = LanguageDetector.load("models/language_classifier.joblib")
result = detector.predict(stt_transcript)
language_for_llm = previous_language if result.label == "uncertain" else result.label
""", language="python")
    st.info("Measure classifier latency separately from STT and end-to-end call latency.")
