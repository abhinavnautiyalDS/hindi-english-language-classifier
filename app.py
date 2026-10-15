from __future__ import annotations
import io, json, time
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.binary_training import LABELS, binary_corpus, evaluate_candidates, normalize_binary_frame, with_other_rejection

ROOT = Path(__file__).resolve().parent
TRAIN_PATH, TEST_PATH = ROOT/"data/processed/train.csv", ROOT/"data/processed/test.csv"
LEGACY_MODEL, LEGACY_EVAL = ROOT/"models/language_classifier.joblib", ROOT/"models/evaluation.json"
MODEL_PATH = ROOT/"models/english_hinglish_classifier.joblib"
st.set_page_config(page_title="English · Hinglish Classifier Lab", page_icon="🗣️", layout="wide")
st.title("English · Hinglish Classifier Lab")
st.caption("English/Hinglish classification with OTHER and UNCERTAIN outcomes for unsupported languages and ambiguous short turns.")
with st.sidebar:
    st.info("Outputs: ENGLISH, HINGLISH, OTHER, or UNCERTAIN. Devanagari (pure Hindi or mixed-script) routes to HINGLISH; other scripts route to OTHER; one-word greetings like hello/bye route to UNCERTAIN.")
    st.write("Model selection: macro-F1 → p95 latency → model size.")
    st.caption("96% is a target, not a guarantee.")
tabs = st.tabs(["Predict", "Label call data", "Train & compare", "Existing baseline", "Integration"])

def saved_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

with tabs[0]:
    model = st.session_state.get("binary_model")
    if model is None:
        model = saved_model()
    utterance = st.text_input("Caller utterance", "Sir after 1 hour")
    if st.button("Classify", type="primary"):
        if model is None:
            st.warning("The binary model is missing. Train a model or wait for the main-branch training workflow.")
        else:
            start = time.perf_counter()
            pred = str(model.predict([utterance])[0])
            ms = (time.perf_counter() - start) * 1000
            a, b = st.columns(2)
            a.metric("Prediction", pred.upper())
            b.metric("Latency", f"{ms:.3f} ms")
            if hasattr(model, "predict_proba"):
                st.bar_chart(pd.Series(dict(zip(model.classes_, model.predict_proba([utterance])[0]))))
            if pred == "uncertain":
                st.info("Ambiguous/neutral utterance: preserve current language; do not switch on a single word.")
            if pred == "other":
                st.info("This utterance was rejected as unsupported/uncertain. It is not being forced into English or Hinglish.")
    upload = st.file_uploader("Batch prediction CSV (text column)", type=["csv"], key="batch")
    if upload and model is not None:
        df = pd.read_csv(upload)
        if "text" not in df:
            st.error("CSV must contain text.")
        else:
            texts = df["text"].fillna("").astype(str)
            df["predicted_label"] = model.predict(texts)
            if hasattr(model, "predict_proba"):
                df["confidence_or_reject_score"] = model.predict_proba(texts).max(axis=1)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download predictions", df.to_csv(index=False).encode(), "predictions.csv", "text/csv")

with tabs[1]:
    st.warning("Verify labels manually. Do not treat heuristic predictions as ground truth.")
    upload = st.file_uploader("Upload caller-turn CSV", type=["csv"], key="labeldata")
    if upload:
        df = pd.read_csv(upload)
        if "text" not in df:
            st.error("CSV must contain text; call_id is recommended.")
        else:
            if "label" not in df:
                df["label"] = ""
            edited = st.data_editor(df, use_container_width=True, num_rows="dynamic",
                column_config={"label": st.column_config.SelectboxColumn("Verified label", options=["", "english", "hinglish", "uncertain", "other"], required=False)})
            st.metric("Verified rows", int(edited["label"].isin(["english", "hinglish", "uncertain", "other"]).sum()))
            st.download_button("Download reviewed labels", edited.to_csv(index=False).encode(), "verified_call_labels.csv", "text/csv")

with tabs[2]:
    st.write("Train the English/Hinglish classifier on the held-out corpus workflow. Verified call labels can be reviewed/exported here; OTHER labels are not silently relabeled as English or Hinglish.")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        st.error("Processed datasets missing. Run scripts/build_dataset.py.")
    else:
        train0, test = binary_corpus(TRAIN_PATH), binary_corpus(TEST_PATH)
        a, b = st.columns(2)
        a.metric("Binary train rows", len(train0))
        b.metric("Held-out test rows", len(test))
        st.dataframe(train0.label.value_counts().rename_axis("label").reset_index(name="rows"))
        calls_upload = st.file_uploader("Optional verified call CSV (text,label)", type=["csv"], key="verifiedcalls")
        if st.button("Train and compare four models", type="primary"):
            try:
                train = train0.copy()
                added = 0
                if calls_upload:
                    calls = pd.read_csv(calls_upload)
                    if not {"text", "label"}.issubset(calls):
                        raise ValueError("Verified call CSV needs text,label.")
                    calls["label"] = calls.label.fillna("").astype(str).str.lower().str.strip()
                    calls = normalize_binary_frame(calls[calls.label.isin(LABELS)].copy())
                    added = len(calls)
                    train = pd.concat([train, calls], ignore_index=True)
                test_text = set(test.text.str.casefold())
                before = len(train)
                train = train[~train.text.str.casefold().isin(test_text)].drop_duplicates("text").reset_index(drop=True)
                removed = before - len(train)
                results, models = evaluate_candidates(train, test)
                table = pd.DataFrame([{"model": n, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "p50_ms": r["latency"]["p50_ms"], "p95_ms": r["latency"]["p95_ms"], "train_seconds": r["train_seconds"], "model_size_mb": r["model_size_mb"], "generalization_gap": r["generalization_gap_accuracy"]} for n, r in results.items()]).sort_values(["macro_f1", "p95_ms"], ascending=[False, True]).reset_index(drop=True)
                best = str(table.iloc[0].model)
                st.session_state["binary_model"] = with_other_rejection(models[best])
                st.session_state["binary_results"] = results
                st.success(f"Best candidate: {best}. Verified binary call rows added: {added}; duplicate/test-overlap rows removed: {removed}.")
                st.dataframe(table, use_container_width=True)
                st.caption("Scores evaluate only English/Hinglish corpus rows, not OTHER-language rejection or production call accuracy.")
                st.dataframe(pd.DataFrame(results[best]["classification_report"]).T)
                st.write("Confusion matrix: rows=actual, columns=predicted")
                st.dataframe(pd.DataFrame(results[best]["confusion_matrix"], index=LABELS, columns=LABELS))
                buf = io.BytesIO()
                joblib.dump(with_other_rejection(models[best]), buf)
                st.download_button("Download selected model", buf.getvalue(), "english_hinglish_classifier.joblib", "application/octet-stream")
                st.download_button("Download metrics", json.dumps({"selected_model": best, "call_rows_added": added, "train_rows": len(train), "test_rows": len(test), "results": results}, indent=2).encode(), "model_comparison.json", "application/json")
            except Exception as e:
                st.error(f"Training failed: {e}")
        if "binary_results" in st.session_state:
            for name, r in st.session_state["binary_results"].items():
                with st.expander(name):
                    x, y, z = st.columns(3)
                    x.metric("Accuracy", f"{r['accuracy']:.2%}")
                    y.metric("Macro-F1", f"{r['macro_f1']:.3f}")
                    z.metric("p95", f"{r['latency']['p95_ms']:.3f} ms")
                    st.dataframe(pd.DataFrame(r["classification_report"]).T)

with tabs[3]:
    if LEGACY_EVAL.exists():
        report = json.loads(LEGACY_EVAL.read_text(encoding="utf-8"))
        st.write("Legacy labels:", report.get("labels", ["hindi", "english", "hinglish"]))
        st.caption("Legacy three-class scores are not directly comparable to binary scores.")
        st.json(report.get("results", {}))
    else:
        st.info("No legacy evaluation report found.")

with tabs[4]:
    st.code('import joblib\nmodel = joblib.load("models/english_hinglish_classifier.joblib")\nlabel = model.predict([finalized_stt_utterance])[0]  # english | hinglish | other | uncertain\n# Update language state before LLM/TTS. For uncertain, preserve current language or clarify.', language="python")
    st.info("Log STT-final, classifier start/end, LLM start/end and TTS start/end separately. Validate OTHER rejection on manually labelled multilingual call turns.")
