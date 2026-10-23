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

## Branch-specific problem and solution

**Problem being addressed:** Classical baseline: fast English/Hinglish classification with OTHER/UNCERTAIN rejection for ambiguous or unsupported input.

**Solution and evidence:** The sections below connect the broader project experiments to this branch. Treat claims as verified only where a dataset archive, model evaluation, committed configuration or changelog supports them; otherwise confirm the exact intervention against this branch's commits before presenting it as historical fact.


## Engineering history: problem → change → evidence

### Why the project started
A voice agent receives speech-to-text (STT) transcripts and must decide which language to use. Script alone is not language: Hindi can be Romanized, English and Hinglish can share Latin script, and short turns such as “hello”, “okay” or “haan ji” may not contain enough evidence to switch.

### Iteration logic
For each model/data iteration, the engineering question is: what failed in the previous attempt, what changed, and what evidence shows whether it helped?

- **Script classifier:** fast script signal, but cannot distinguish English from Romanized Hindi or resolve code-mixing. It remains a separate task, not a universal language detector.
- **Classical English/Hinglish baseline:** TF-IDF character n-grams + LinearSVC with OTHER/UNCERTAIN rejection. Its saved benchmark reports 9,389 test rows, 97.2628% accuracy, 97.0005% macro-F1, p50 ≈ 1.337 ms and p95 ≈ 1.509 ms. This binary benchmark is not directly comparable to the later 12-label benchmark.
- **Qwen + LoRA experiment:** explored contextual/generative classification. LoRA trains small adapter matrices while most base weights stay frozen; inference still needs the base Qwen model. Exact checkpoint, parameter count and LoRA settings must come from the committed config/logs, not be guessed from the family name.
- **BERT-Tiny E2/E4:** moved toward compact fixed-label classification for a CPU-constrained pipeline. The current E2 v4.4 config uses 2 transformer layers, hidden size 128, 2 attention heads, intermediate size 512, vocabulary 30,522 and 12 output labels.
- **Version 4 data audit:** the v4 notes report removing 9,992 of 10,000 synthetic Punjabi rows because the samples were mostly Hindi-looking Devanagari text with Punjabi labels. Repetitive templates were trimmed, STT-like noise was added and group-aware splits were rebuilt.
- **Versions 4.1–4.3:** treat archive sizes as distinct snapshots. Where a branch has no build report/changelog, describe the exact purpose as a hypothesis and verify it against commits before claiming it as historical fact.
- **Version 4.4:** targets false switches on short acknowledgements. It adds 15,783 acknowledgement rows to none and relabels 294 existing rows to none, including native/Romanized forms and case/punctuation variants. “Okay” alone is neutral; “Okay, I want to cancel my order” contains language-bearing evidence.

### Current multilingual snapshot
The uploaded v4.4 dataset has 903,414 rows, 12 labels and 141 categories: 723,240 train / 90,066 validation / 90,108 test. The uploaded BERT-Tiny E2 v4.4 report records 99.868% accuracy and 99.868% macro-F1 on the prepared test split; none has the lowest reported class F1 at about 99.446%. These are offline prepared-test results, not a guarantee for live STT calls.

### Data sources and model references
- [COMI-LINGUA](https://github.com/lingo-iitgn/COMI-LINGUA) — human-annotated Hindi/English/Hinglish source.
- [PHINC](https://huggingface.co/datasets/LingoIITGN/PHINC) — Romanized Hinglish and English translations.
- [Tatoeba English–Hindi selection](https://www.manythings.org/anki/hin-eng.zip) — paired monolingual examples.
- [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID) — Indic language-identification reference.
- [Google BERT-Tiny base checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
- [Sarvam Text Language Detection API](https://docs.sarvam.ai/api-reference/text/identify-language) — hosted competitor/reference, not a downloadable open-source model.
- [fastText language identification](https://fasttext.cc/docs/en/language-identification.html) · [CLD3](https://github.com/google/cld3) · [Lingua](https://github.com/pemistahl/lingua).

### Open-source release gates
Before release, publish: (1) accuracy/precision/recall/macro-F1, confusion matrix, calibration and failure analysis; (2) leakage/near-duplicate checks, annotation audit and independent unseen data; (3) CPU p50/p90/p95, throughput, RAM, cold/warm start and concurrency; (4) noisy, misspelled, short/long, mixed-script, unsupported-language and malformed-Unicode tests; (5) same-sample competitor benchmarks; (6) API validation, timeouts, concurrency and rollback; (7) dataset/model/dependency license, attribution, privacy and hostile-input checks; and (8) model card, reproducible evaluation, pinned install, known limitations and versioned artifacts. Report Sarvam's network-inclusive latency separately from local CPU inference.

