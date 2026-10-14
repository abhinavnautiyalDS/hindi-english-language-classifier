from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS = ("hindi", "english", "hinglish")
PURE_LANGUAGE_THRESHOLD = 0.80

COMI_FILES = {
    "train": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/LID_train.csv?download=true",
    "test": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/LID_test.csv?download=true",
}
PHINC_URL = "https://huggingface.co/datasets/LingoIITGN/PHINC/resolve/main/PHINC.csv?download=true"
COMI_NAME = "COMI-LINGUA (Lingo Research Group, IIT Gandhinagar)"
COMI_LICENSE = "CC BY-SA 4.0 (conservative interpretation; upstream GitHub repository declares CC-BY-SA-4.0)"
PHINC_NAME = "PHINC (Parallel Hinglish Social Media Code-Mixed Corpus)"
PHINC_LICENSE = "CC BY 4.0"
COMI_URL = "https://github.com/lingo-iitgn/COMI-LINGUA"
PHINC_SOURCE_URL = "https://huggingface.co/datasets/LingoIITGN/PHINC"


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 1000:
        print(f"Using cached source: {target} ({target.stat().st_size:,} bytes)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temp.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temp.replace(target)
    print(f"Downloaded {target} ({target.stat().st_size:,} bytes)")


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def choose_comi_columns(frame: pd.DataFrame) -> tuple[str, list[str], str]:
    normalized = {col: re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in frame.columns}
    text_candidates = [
        col for col, name in normalized.items()
        if any(key in name for key in ("sentence", "utterance", "text", "transcript"))
    ]
    if not text_candidates:
        raise ValueError(f"Could not identify COMI-LINGUA sentence column. Found: {list(frame.columns)}")
    text_col = sorted(text_candidates, key=lambda c: (("sentence" not in normalized[c]), len(normalized[c])))[0]
    annotator_cols = [
        col for col, name in normalized.items()
        if "annotator" in name and "annotatedby" in name
    ]
    if len(annotator_cols) >= 2:
        return text_col, annotator_cols, "majority vote across human annotators"
    tag_candidates = [
        col for col, name in normalized.items()
        if any(key in name for key in ("lid", "languagetag", "languagelabel", "tags", "label"))
        and col != text_col
    ]
    if not tag_candidates:
        raise ValueError(f"Could not identify COMI-LINGUA language-tag column. Found: {list(frame.columns)}")
    tag_col = sorted(tag_candidates, key=lambda c: (("lid" not in normalized[c] and "tag" not in normalized[c]), len(normalized[c])))[0]
    return text_col, [tag_col], "single available language-tag column"


def sentence_label(tags_value: object) -> str | None:
    tags = re.findall(r"(?<![a-z])(hi|en|ot|other)(?![a-z])", str(tags_value).lower())
    hi_count = tags.count("hi")
    en_count = tags.count("en")
    total = hi_count + en_count
    if total == 0:
        return None
    hi_ratio = hi_count / total
    en_ratio = en_count / total
    if hi_ratio >= PURE_LANGUAGE_THRESHOLD:
        return "hindi"
    if en_ratio >= PURE_LANGUAGE_THRESHOLD:
        return "english"
    return "hinglish"


def majority_sentence_label(annotation_values: list[object]) -> str | None:
    labels = [label for value in annotation_values if (label := sentence_label(value)) is not None]
    if not labels:
        return None
    counts = Counter(labels)
    label, votes = counts.most_common(1)[0]
    if len(annotation_values) == 1:
        return label
    if votes > len(labels) / 2:
        return label
    return None


def convert_comi_split(split: str, raw_path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(raw_path, low_memory=False)
    text_col, tag_cols, annotation_method = choose_comi_columns(frame)
    print(f"{split}: COMI columns = {list(frame.columns)}")
    print(f"{split}: labels from {tag_cols} ({annotation_method})")
    data = pd.DataFrame({
        "text": frame[text_col].map(normalize_text),
        "label": frame[tag_cols].apply(lambda row: majority_sentence_label(row.tolist()), axis=1),
        "source": "COMI-LINGUA",
    })
    before = len(data)
    data = data.dropna(subset=["label"])
    data = data[data["text"].str.len() >= 2]
    after_label_filter = len(data)
    data["text_key"] = data["text"].str.casefold()
    data = data.drop_duplicates(subset=["text_key"], keep="first")
    data = data[["text", "label", "source"]].reset_index(drop=True)
    audit = {
        "input_rows": before,
        "rows_with_usable_majority_labels": after_label_filter,
        "rows_after_exact_casefold_deduplication": len(data),
        "dropped_rows": before - len(data),
        "label_counts": {label: int((data["label"] == label).sum()) for label in LABELS},
        "text_column": text_col,
        "tag_columns": tag_cols,
        "annotation_method": annotation_method,
    }
    return data, audit


def build_phinc_splits(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = pd.read_csv(raw_path, low_memory=False)
    normalized = {col: re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in frame.columns}
    english_candidates = [
        col for col, name in normalized.items()
        if "english" in name and ("translation" in name or "translated" in name)
    ]
    if not english_candidates:
        english_candidates = [col for col, name in normalized.items() if "english" in name]
    mixed_candidates = [
        col for col, name in normalized.items()
        if col not in english_candidates and any(key in name for key in ("hinglish", "sentence", "codemixed", "mixed"))
    ]
    if not english_candidates or not mixed_candidates:
        raise ValueError(f"Could not identify PHINC Hinglish/English columns. Found: {list(frame.columns)}")
    english_col = english_candidates[0]
    mixed_col = mixed_candidates[0]

    pairs = pd.DataFrame({
        "hinglish": frame[mixed_col].map(normalize_text),
        "english": frame[english_col].map(normalize_text),
    })
    pairs = pairs[(pairs["hinglish"].str.len() >= 2) & (pairs["english"].str.len() >= 2)]
    pairs["pair_key"] = pairs["hinglish"].str.casefold() + "\u241f" + pairs["english"].str.casefold()
    pairs = pairs.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"]).reset_index(drop=True)

    order = np.random.default_rng(42).permutation(len(pairs))
    cut = int(round(len(pairs) * 0.80))
    split_indices = {"train": order[:cut], "test": order[cut:]}
    result = {}
    for split, indices in split_indices.items():
        selected = pairs.iloc[indices]
        rows = []
        for row in selected.itertuples(index=False):
            rows.append({"text": row.hinglish, "label": "hinglish", "source": "PHINC"})
            rows.append({"text": row.english, "label": "english", "source": "PHINC"})
        result[split] = pd.DataFrame(rows, columns=["text", "label", "source"])
    audit = {
        "dataset": PHINC_NAME,
        "source_url": PHINC_SOURCE_URL,
        "license": PHINC_LICENSE,
        "input_rows": len(frame),
        "clean_unique_pairs": len(pairs),
        "text_column": mixed_col,
        "english_translation_column": english_col,
        "split_rule": "deterministic 80/20 split by paired row; both members of each translation pair stay in the same split",
        "label_counts": {
            split: {label: int((df["label"] == label).sum()) for label in LABELS}
            for split, df in result.items()
        },
    }
    return result["train"], result["test"], audit


def clean_combined(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["text"] = frame["text"].map(normalize_text)
    frame["label"] = frame["label"].astype(str).str.lower()
    frame = frame[(frame["text"].str.len() >= 2) & frame["label"].isin(LABELS)]
    frame["_key"] = frame["text"].str.casefold()
    frame = frame.drop_duplicates(subset=["_key"], keep="first")
    return frame.drop(columns=["_key"]).reset_index(drop=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    comi_frames, comi_audit = {}, {}

    for split, url in COMI_FILES.items():
        raw_path = RAW_DIR / f"COMI-LINGUA_LID_{split}.csv"
        download(url, raw_path)
        comi_frames[split], comi_audit[split] = convert_comi_split(split, raw_path)

    phinc_path = RAW_DIR / "PHINC.csv"
    download(PHINC_URL, phinc_path)
    phinc_train, phinc_test, phinc_audit = build_phinc_splits(phinc_path)

    frames = {
        "train": clean_combined(pd.concat([comi_frames["train"], phinc_train], ignore_index=True)),
        "test": clean_combined(pd.concat([comi_frames["test"], phinc_test], ignore_index=True)),
    }
    test_keys = set(frames["test"]["text"].str.casefold())
    before = len(frames["train"])
    frames["train"] = frames["train"][
        ~frames["train"]["text"].str.casefold().isin(test_keys)
    ].reset_index(drop=True)

    audit = {
        "dataset": "COMI-LINGUA + PHINC",
        "source_url": COMI_URL,
        "sources": [
            {
                "name": COMI_NAME,
                "url": COMI_URL,
                "license": COMI_LICENSE,
                "use": "official LID train/test splits; majority vote over human annotator tags",
            },
            {
                "name": PHINC_NAME,
                "url": PHINC_SOURCE_URL,
                "license": PHINC_LICENSE,
                "use": "Hinglish source text and human English translations; paired rows split together",
            },
        ],
        "license": "COMI-LINGUA: conservatively treated as CC BY-SA 4.0; PHINC: CC BY 4.0. See DATASET_CARD.md.",
        "label_rule": f"COMI per-annotator: hi ratio >= {PURE_LANGUAGE_THRESHOLD:.0%} => hindi; en ratio >= {PURE_LANGUAGE_THRESHOLD:.0%} => english; otherwise hinglish; majority vote across human annotators; PHINC source text => hinglish and human translation => english",
        "cleaning": [
            "Unicode NFKC normalization",
            "remove zero-width/BOM characters and collapse whitespace",
            "majority vote across human COMI-LINGUA annotators; discard ties",
            "drop empty/very short text",
            "deduplicate exact case-insensitive text within each combined split",
            "split PHINC by translation pair before expanding to two labeled examples",
            "remove exact train/test text overlap across all sources",
        ],
        "comi_splits": comi_audit,
        "phinc": phinc_audit,
        "train_test_overlap_removed": before - len(frames["train"]),
        "final_label_counts": {
            split: {label: int((df["label"] == label).sum()) for label in LABELS}
            for split, df in frames.items()
        },
        "final_source_counts": {
            split: {source: int((df["source"] == source).sum()) for source in df["source"].unique()}
            for split, df in frames.items()
        },
    }

    for split, df in frames.items():
        if df.empty or set(df["label"]) != set(LABELS):
            raise ValueError(
                f"{split} split does not contain all three classes after cleaning. "
                f"Counts: {audit['final_label_counts'][split]}"
            )
        df.to_csv(PROCESSED_DIR / f"{split}.csv", index=False, encoding="utf-8")

    (PROCESSED_DIR / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Final class counts:", json.dumps(audit["final_label_counts"], indent=2))
    print("Final source counts:", json.dumps(audit["final_source_counts"], indent=2))
    print(f"Clean train/test files written to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
