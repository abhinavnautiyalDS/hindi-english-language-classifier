# English / Hinglish Language Classifier

Lightweight CPU classifier for finalized voice-agent STT utterances. The binary model distinguishes English from Hinglish and has explicit **OTHER** and **UNCERTAIN** outcomes so unsupported inputs are not always forced into one of the two classes.

## Important behavior

The trained text model is binary: `english` and `hinglish`. A rejection wrapper adds `other`:
- Devanagari text (pure Hindi or mixed Hindi-English) returns `hinglish` per product policy.
- Romanized Hindi cues return `hinglish`.
- One-word greetings such as `hello`, `bye`, `thanks`, `okay` return `uncertain`; preserve current response language.
- Unsupported non-Latin scripts such as Telugu-only text return `other`.
- Low-margin Latin-script predictions return `uncertain`, not `other`.
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
label = model.predict([finalized_stt_utterance])[0]  # english | hinglish | other | uncertain
```

Update language state per turn, not only at call start. For `other`, use an unsupported-language fallback. For `uncertain`, preserve the current voice-agent language or ask for clarification; do not switch on a one-word acknowledgement. Log STT-final, classifier start/end, LLM start/end and TTS start/end to separate classifier errors from orchestration lag and network latency.

## Data sources

- COMI-LINGUA: https://aclanthology.org/2025.findings-emnlp.422/
- PHINC: https://aclanthology.org/2020.wnut-1.7/
- Tatoeba English-Hindi pairs: https://mail.manythings.org/anki/hin-eng.zip

The training corpus is not a substitute for a representative, manually labelled multilingual call test set.

---

## Branch focus: problem → change → evidence

**Branch-specific purpose:** BERT-Tiny experiment: test a compact contextual model as a step beyond sparse TF-IDF features; verify exact checkpoint and labels from committed config.


## Problem → solution → evidence: project evolution

### Product problem
A real-time voice agent uses STT transcripts to decide the language for the next response. Script alone is insufficient: Hindi can be Romanized, English and Hinglish can share Latin script, and a short “hello”, “okay” or “haan ji” may not justify switching.

### Experiments
- **Script classifier:** BERT-Tiny predicts DEVANAGARI vs LATIN. Useful as a script signal, but not a language detector.
- **Classical English/Hinglish:** TF-IDF character n-grams + LinearSVC with OTHER/UNCERTAIN rejection. Saved benchmark: 9,389 test rows, 97.2628% accuracy, 97.0005% macro-F1, p50 ≈ 1.337 ms, p95 ≈ 1.509 ms. Binary benchmark only; not comparable directly to 12-label results.
- **Qwen + LoRA:** explored context-aware classification. LoRA reduces trainable parameters, but inference still needs the base Qwen checkpoint. Exact parameter count, checkpoint and LoRA rank/alpha must be read from experiment config—not guessed.
- **BERT-Tiny E2/E4:** compact fixed-label model for a CPU-constrained use case. Current E2 v4.4 architecture: 2 layers, hidden size 128, 2 attention heads, intermediate size 512, vocab 30,522, 12 labels.
- **Version 4 data audit:** removed 9,992/10,000 flagged synthetic Punjabi rows reported to be mostly Hindi-looking Devanagari with Punjabi labels; trimmed repetitive templates, added STT-like noise and rebuilt group-aware splits.
- **Versions 4.1–4.3:** treat each archive as a separate snapshot. If the branch lacks a build report, state only verified counts and verify detailed historical changes against commits before claiming them as fact.
- **Version 4.4:** added 15,783 acknowledgement rows to none and relabeled 294 existing rows to none. Includes native/Romanized forms and case/punctuation variants. “Okay” alone is neutral; “Okay, I want to cancel my order” contains language-bearing evidence.

### Current v4.4 result (multilingual main project)
903,414 rows; 12 labels; 141 categories; train 723,240 / validation 90,066 / test 90,108. Uploaded BERT-Tiny E2 report: 99.868% accuracy and 99.868% macro-F1 on the prepared test split. The none class has lowest F1 (~99.446%). This is not proof of the same performance on live STT calls.

### Clickable references
- [COMI-LINGUA](https://github.com/lingo-iitgn/COMI-LINGUA) · [PHINC](https://huggingface.co/datasets/LingoIITGN/PHINC) · [Tatoeba English–Hindi](https://www.manythings.org/anki/hin-eng.zip)
- [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID)
- [Google BERT-Tiny checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) · [Qwen models](https://huggingface.co/Qwen)
- [Sarvam text language detection API](https://docs.sarvam.ai/api-reference/text/identify-language) · [fastText LID](https://fasttext.cc/docs/en/language-identification.html) · [CLD3](https://github.com/google/cld3) · [Lingua](https://github.com/pemistahl/lingua)

### Open-source release gates
Publish model quality (per-class metrics, confusion matrix, calibration and failures); dataset integrity (leakage, near duplicates, annotation and unseen data); CPU efficiency (p50/p90/p95, throughput, RAM, cold/warm start, concurrency); robustness (noise, typos, short/long text, mixed scripts, unsupported languages, malformed Unicode); fair same-sample benchmarks; API/timeouts/concurrency/rollback tests; dataset/model/dependency licensing, attribution and privacy checks; and a model card with reproducible evaluation, pinned install and known limitations. Separate hosted Sarvam network latency from local CPU inference.

