from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.language_classifier import DEFAULT_MODEL_PATH, LanguageDetector

app = FastAPI(title="Hindi-English Language Classifier", version="0.1.0")
detector = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


@app.on_event("startup")
def load_detector():
    global detector
    if Path(DEFAULT_MODEL_PATH).exists():
        detector = LanguageDetector.load(DEFAULT_MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict")
def predict(request: PredictRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model missing. Run python train.py first.")
    return detector.predict(request.text).to_dict()
