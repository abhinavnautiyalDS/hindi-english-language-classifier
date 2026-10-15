from __future__ import annotations
import io, re, time, unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

LABELS = ("english", "hinglish")
OUTPUT_LABELS = ("english", "hinglish", "other", "uncertain")

# Common Romanized Hindi cues. This is an explicit heuristic layer for short
# turns; expand it using reviewed call examples rather than auto-labeling them.
ROMANIZED_HINDI_CUES = set("""
hai hain ho hun hoga hogi honge tha thi mujhe mera meri mere aap apko aapko
aapka aapki tum tumhe tu kya kaise kaisa kaunsi kaun kab kahan kaha kyun kyu kyon
nahi nahin na nhi haan han ji jii achha acha accha thik theek kr do karo karna
chahiye chahte chahta bata batao bataiye bataye bol raha rahi abhi kal aaj par
se ko ka ki ke mein mai mey main hum ham hamara tumhara yeh ye woh vo hai
hain namaste dhanyawad shukriya yahan wahan kitna kitne kitni please batao
""".split())
SINGLE_WORD_UNCERTAIN = {"hello", "hi", "hey", "bye", "goodbye", "thanks", "okay", "ok", "yes", "no", "sure", "please", "right", "correct", "great", "fine"}
DEVANAGARI_RANGE = re.compile(r"[\\u0900-\\u097F]")
NON_LATIN_ALPHA = re.compile(r"[^\\W\\d_]", re.UNICODE)


def script_profile(text: str) -> tuple[bool, bool, bool]:
    """Return (has_devanagari, has_other_non_latin, has_latin_letters)."""
    has_devanagari = has_other_non_latin = has_latin = False
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            has_latin = True
        elif "DEVANAGARI" in name:
            has_devanagari = True
        else:
            has_other_non_latin = True
    return has_devanagari, has_other_non_latin, has_latin


def romanized_hindi_cue(text: str) -> bool:
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())
    # A single cue can be enough for a short mixed utterance ("haan please").
    return any(token in ROMANIZED_HINDI_CUES for token in tokens)


class OtherLanguageRejector:
    """English/Hinglish model with deterministic script/cue rules and abstention.

    Policy: Devanagari (pure Hindi or mixed script) => Hinglish; other non-Latin
    scripts without Devanagari => OTHER; Romanized Hindi cues => Hinglish;
    one-word English greetings/acknowledgements => UNCERTAIN; low-margin Latin
    text => UNCERTAIN. Remaining Latin text uses the trained binary classifier.
    """
    def __init__(self, model, min_margin: float = 0.35):
        self.model = model
        self.min_margin = float(min_margin)
        self.classes_ = np.array(OUTPUT_LABELS, dtype=object)

    def _route(self, text):
        cleaned = str(text or "").strip()
        has_devanagari, has_other_non_latin, has_latin = script_profile(cleaned)
        if not cleaned:
            return "uncertain", None
        if has_devanagari:
            return "hinglish", None
        if has_other_non_latin and not has_latin:
            return "other", None
        if has_other_non_latin and has_latin:
            return "other", None
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", cleaned.lower())
        if cleaned.lower() in {"okay sir", "ok sir", "yes sir", "no sir"}:
            return "uncertain", None
        if len(tokens) == 1 and tokens[0] in SINGLE_WORD_UNCERTAIN:
            return "uncertain", None
        if romanized_hindi_cue(cleaned):
            return "hinglish", None
        if not has_latin:
            return "uncertain", None
        if len(tokens) < 2 or sum(ch.isalpha() for ch in cleaned) < 5:
            return "uncertain", None
        score = float(np.asarray(self.model.decision_function([cleaned])).reshape(-1)[0])
        if abs(score) < self.min_margin:
            return "uncertain", score
        return str(self.model.classes_[1] if score > 0 else self.model.classes_[0]), score

    def predict(self, texts):
        return np.asarray([self._route(text)[0] for text in texts], dtype=object)

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            label, score = self._route(text)
            if label == "hinglish" and score is None:
                rows.append([0.0, 1.0, 0.0, 0.0])
            elif label == "other":
                rows.append([0.0, 0.0, 1.0, 0.0])
            elif label == "uncertain":
                rows.append([0.0, 0.0, 0.0, 1.0])
            else:
                strength = 1.0 / (1.0 + np.exp(-min(abs(float(score)), 30.0)))
                rows.append([1.0-strength, strength, 0.0, 0.0] if score > 0 else [strength, 1.0-strength, 0.0, 0.0])
        return np.asarray(rows, dtype=float)


def with_other_rejection(model, min_margin: float = 0.35):
    return OtherLanguageRejector(model, min_margin=min_margin)

