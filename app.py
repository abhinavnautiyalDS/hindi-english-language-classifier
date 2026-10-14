from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.language_classifier import (
    LABELS, LanguageDetector, build_logistic_regression, build_linear_svm, load_dataset,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample.csv"
MODEL_PATH = ROOT / "models" / "language_classifier.joblib"

st.set_page_config(page_title="Hindi-English Language Classifier", page_icon="🗣️", layout="wide")
st.title("Hindi · English · Hinglish Classifier")
st.caption("Low-latency language detection for voice-agent STT transcripts.")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Training CSV path", str(DATA_PATH))
    min_confidence = st.slider("Minimum confidence", 0.30, 0.95, 0.55, 0.05)
    min_chars = st.slider("Minimum characters", 1, 12, 4)

@st.cache_resource
def get_model(model_path, csv_path):
    if Path(model_path).exists():
        import joblib
        return joblib.load(model_path), "Bundled trained model"
    data = load_dataset(csv_path)
    model = build_logistic_regression()
    model.fit(data["text"], data["label"])
    return model, "On-the-fly demo model"

try:
    model, model_source = get_model(str(MODEL_PATH), data_path)
    detector = LanguageDetector(model, min_confidence=min_confidence, min_chars=min_chars)
except Exception as exc:
    st.error(f"Unable to load model or dataset: {exc}")
    st.stop()

st.caption(f"Model source: {model_source}")
single, batch, evaluate, integration = st.tabs(
    ["Single transcript", "Batch CSV", "Evaluate models", "Integration"])

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
            st.error("CSV must contain a 'text' column.")
        else:
            predictions = [detector.predict(str(t)) for t in df["text"].fillna("")]
            df["predicted_label"] = [p.label for p in predictions]
            df["confidence"] = [p.confidence for p in predictions]
            df["latency_ms"] = [p.latency_ms for p in predictions]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download predictions", df.to_csv(index=False).encode("utf-8"),
                               "language_predictions.csv", "text/csv")
            if "label" in df:
                valid = df["label"].astype(str).str.lower().isin(LABELS)
                if valid.any():
                    st.metric("Accuracy", f"{accuracy_score(df.loc[valid, 'label'].str.lower(), df.loc[valid, 'predicted_label']):.1%}")

with evaluate:
    st.write("Compares Logistic Regression and Linear SVM with a stratified holdout split.")
    if st.button("Run model comparison"):
        try:
            data = load_dataset(data_path)
            if data["label"].value_counts().min() < 2:
                st.error("Need at least two examples per class.")
            else:
                tr, te = train_test_split(data, test_size=0.30, random_state=42, stratify=data["label"])
                summaries = []
                for name, builder in [("Logistic Regression", build_logistic_regression),
                                      ("Linear SVM", build_linear_svm)]:
                    candidate = builder()
                    candidate.fit(tr["text"], tr["label"])
                    pred = candidate.predict(te["text"])
                    summaries.append({"model": name,
                                      "accuracy": accuracy_score(te["label"], pred),
                                      "macro_f1": f1_score(te["label"], pred, average="macro", zero_division=0)})
                    st.markdown(f"**{name} — per-class report**")
                    st.dataframe(pd.DataFrame(classification_report(
                        te["label"], pred, labels=list(LABELS), output_dict=True, zero_division=0)).T)
                    st.write("Confusion matrix (rows = true; columns = predicted)")
                    st.dataframe(pd.DataFrame(confusion_matrix(te["label"], pred, labels=list(LABELS)),
                                              index=LABELS, columns=LABELS))
                st.dataframe(pd.DataFrame(summaries), use_container_width=True)
                st.warning("Sample-data results are a smoke test only, not production estimates.")
        except Exception as exc:
            st.error(str(exc))

with integration:
    st.code('''from src.language_classifier import LanguageDetector

detector = LanguageDetector.load("models/language_classifier.joblib")
result = detector.predict(stt_transcript)
language_for_llm = previous_language if result.label == "uncertain" else result.label
''', language="python")
    st.info("Measure classifier latency separately from STT and end-to-end call latency.")
