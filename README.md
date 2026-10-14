# Hindi-English Language Classifier

Low-latency text classification for Hindi, English, and Hinglish transcripts from voice-agent STT.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Dataset format
CSV columns: `text,label`. Allowed labels: `hindi`, `english`, `hinglish`.

The bundled sample dataset is only a smoke-test fixture. Any model trained on it is not production-ready and must not be used to claim real-world accuracy. Replace it with representative, consented, speaker-separated transcripts from the target STT engine.

## Evaluation
The training script compares word + character TF-IDF features with Logistic Regression and Linear SVM, using accuracy, macro-F1, per-class metrics, confusion matrix, and p50/p95 classifier inference latency. This excludes STT and end-to-end voice latency.

## Streamlit Cloud
Choose this repo, branch `main`, and app entrypoint `app.py` in Streamlit Community Cloud. `requirements.txt` must be at repository root.

## Production notes
A text classifier cannot directly detect acoustic accent. Test with the exact STT outputs used in production. For short or ambiguous turns, preserve the previous conversation language rather than switching based on one token.