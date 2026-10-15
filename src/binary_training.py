from __future__ import annotations
import io, time
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

LABELS=("english","hinglish")

def normalize_binary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"text","label"}.issubset(frame.columns):
        raise ValueError("CSV must contain text and label columns.")
    data=frame.copy()
    data["text"]=data["text"].fillna("").astype(str).str.lower().str.replace(r"\s+"," ",regex=True).str.strip()
    data["label"]=data["label"].fillna("").astype(str).str.strip().str.lower()
    aliases={"eng":"english","en":"english","mixed":"hinglish"}
    data["label"]=data["label"].map(lambda x:aliases.get(x,x))
    data=data[(data["text"].str.len()>=2)&data["label"].isin(LABELS)].copy()
    if data["label"].nunique()!=2:
        raise ValueError("Need verified examples for both labels: english and hinglish.")
    return data.drop_duplicates(subset=["text"],keep="first").reset_index(drop=True)

def binary_corpus(path: str | Path) -> pd.DataFrame:
    frame=pd.read_csv(path)
    if not {"text","label"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain text,label columns.")
    frame["label"]=frame["label"].astype(str).str.strip().str.lower()
    return normalize_binary_frame(frame[frame["label"].isin(LABELS)].copy())

def make_candidates():
    def features():
        return FeatureUnion([
            ("word",TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=2,max_features=100_000,sublinear_tf=True)),
            ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,max_features=150_000,sublinear_tf=True)),
        ])
    return {
        "logistic_regression":Pipeline([("features",features()),("classifier",LogisticRegression(max_iter=1500,class_weight="balanced",C=1.0,random_state=42))]),
        "linear_svm":Pipeline([("features",features()),("classifier",LinearSVC(class_weight="balanced",C=1.0,random_state=42))]),
        "sgd_log_loss":Pipeline([("features",features()),("classifier",SGDClassifier(loss="log_loss",alpha=1e-5,max_iter=1500,class_weight="balanced",random_state=42,tol=1e-3))]),
        "complement_nb":Pipeline([("features",features()),("classifier",ComplementNB(alpha=0.5))]),
    }

def benchmark_latency(model,texts,repeats=3,max_examples=500):
    samples=list(texts[:max_examples])
    if not samples: return {"p50_ms":0.0,"p95_ms":0.0,"n_calls":0}
    model.predict([samples[0]]); timings=[]
    for _ in range(repeats):
        for text in samples:
            start=time.perf_counter(); model.predict([text]); timings.append((time.perf_counter()-start)*1000)
    return {"p50_ms":float(np.percentile(timings,50)),"p95_ms":float(np.percentile(timings,95)),"n_calls":len(timings)}

def evaluate_candidates(train:pd.DataFrame,test:pd.DataFrame):
    if set(train["label"])!=set(LABELS) or set(test["label"])!=set(LABELS):
        raise ValueError("Both train and test must contain English and Hinglish.")
    overlap=set(train["text"].str.casefold()) & set(test["text"].str.casefold())
    if overlap: raise ValueError(f"Leakage check failed: {len(overlap)} exact texts overlap train/test.")
    results,fitted={},{}
    for name,model in make_candidates().items():
        start=time.perf_counter(); model.fit(train["text"],train["label"]); train_seconds=time.perf_counter()-start
        train_pred=model.predict(train["text"]); test_pred=model.predict(test["text"])
        buf=io.BytesIO(); joblib.dump(model,buf); size_mb=len(buf.getvalue())/(1024*1024)
        train_acc=accuracy_score(train["label"],train_pred); test_acc=accuracy_score(test["label"],test_pred)
        results[name]={
            "accuracy":float(test_acc),
            "macro_f1":float(f1_score(test["label"],test_pred,labels=list(LABELS),average="macro",zero_division=0)),
            "train_accuracy":float(train_acc),
            "train_macro_f1":float(f1_score(train["label"],train_pred,labels=list(LABELS),average="macro",zero_division=0)),
            "generalization_gap_accuracy":float(train_acc-test_acc),
            "train_seconds":train_seconds,"model_size_mb":size_mb,
            "latency":benchmark_latency(model,test["text"].tolist()),
            "classification_report":classification_report(test["label"],test_pred,labels=list(LABELS),output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(test["label"],test_pred,labels=list(LABELS)).tolist(),
            "test_rows":len(test),
        }
        fitted[name]=model
    return results,fitted
