# English / Hinglish Language Classifier

Lightweight CPU classifier for finalized voice-agent STT utterances. The binary model distinguishes English from Hinglish and has an explicit **OTHER** rejection outcome so unsupported inputs are not always forced into one of the two classes.

## Important behavior

The trained text model is binary: `english` and `hinglish`. A rejection wrapper adds `other`:
- Utterances containing letters from a non-Latin script (for example Telugu or Devanagari/Hindi) return `other`.
- Latin-script utterances whose LinearSVC decision margin is below 0.35 return `other`.
- Other confident Latin-script text can still be confused with English/Hinglish. This is not universal language identification; add manually labelled examples from the actual languages you need to reject and evaluate them separately.

**Do not interpret OTHER as Hindi.** Your voice-agent orchestration should handle it explicitly (for example, ask the caller to repeat in English/Hinglish or route to a multilingual path) rather than silently changing it to Hindi.

## Why the previous behavior happened

The earlier classifier had only two labels, English and Hinglish. A binary classifier must choose one of them for any input, even Telugu, Hindi, or unrelated text. That is why Telugu-script input could be returned as ENGLISH. The new wrapper rejects non-Latin scripts and low-margin inputs as OTHER, but it does not replace a real multilingual language-identification dataset.

## Streamlit

The app is `app.py` on `main`:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Tabs include prediction/batch CSV, manually verified call-label review, model comparison, legacy baseline and integration notes. Do not treat model predictions as ground truth when reviewing transcripts.

## Training and evaluation

```bash
python scripts/build_dataset.py
python -m pytest -q
python train.py
python train_binary.py
# Add manually verified call labels without replacing original datasets:
python train_binary.py --call-data /path/to/verified_call_data.csv
```

The binary training script compares Logistic Regression, Linear SVM, SGD Log Loss and Complement Naive Bayes, then persists the top candidate behind the OTHER rejection wrapper. Corpus metrics cover English/Hinglish examples only; they do not measure OTHER detection.

Latest evaluation report: [models/binary_evaluation.json](models/binary_evaluation.json). Public-corpus accuracy is not production-call accuracy. Use manually verified, consented, anonymized, speaker/call-separated STT transcripts for validation. Include Telugu, Hindi, and any other languages expected in deployment as explicit OTHER test cases.

## Voice-agent integration

Load `models/english_hinglish_classifier.joblib` and classify each **finalized caller turn** before selecting the response language:
```python
import joblib
model = joblib.load("models/english_hinglish_classifier.joblib")
label = model.predict([finalized_stt_utterance])[0]  # english | hinglish | other
```

Update language state per turn, not only at call start. For `other`, do not map to Hindi or Hinglish; follow a deliberate fallback. Log STT-final, classifier start/end, LLM start/end and TTS start/end to separate classifier errors from orchestration lag and network latency.

## Data sources

- COMI-LINGUA: https://aclanthology.org/2025.findings-emnlp.422/
- PHINC: https://aclanthology.org/2020.wnut-1.7/
- Tatoeba English-Hindi pairs: https://mail.manythings.org/anki/hin-eng.zip

The training corpus is not a substitute for a representative, manually labelled multilingual call test set.
