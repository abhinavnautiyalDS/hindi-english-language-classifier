from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import zipfile
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
TATOEBA_URL = "https://mail.manythings.org/anki/hin-eng.zip"

COMI_NAME = "COMI-LINGUA (Lingo Research Group, IIT Gandhinagar)"
COMI_LICENSE = "CC BY-SA 4.0 (conservative interpretation; upstream GitHub repository declares CC-BY-SA-4.0)"
PHINC_NAME = "PHINC (Parallel Hinglish Social Media Code-Mixed Corpus)"
PHINC_LICENSE = "CC BY 4.0"
TATOEBA_NAME = "Tatoeba English-Hindi sentence pairs (ManyThings.org selection)"
TATOEBA_LICENSE = "CC BY 2.0 France"
COMI_URL = "https://github.com/lingo-iitgn/COMI-LINGUA"
PHINC_SOURCE_URL = "https://huggingface.co/datasets/LingoIITGN/PHINC"
TATOEBA_SOURCE_URL = "https://www.manythings.org/anki/hin-eng.zip"


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 1000:
        print(f"Using cached source: {target} ({target.stat().st_size:,} bytes)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=(30, 180), headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36", "Accept": "application/zip,application/octet-stream,*/*"}) as response:
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
    return label if votes > len(labels) / 2 else None


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
    english_col, mixed_col = english_candidates[0], mixed_candidates[0]
    pairs = pd.DataFrame({
        "hinglish": frame[mixed_col].map(normalize_text),
        "english": frame[english_col].map(normalize_text),
    })
    pairs = pairs[(pairs["hinglish"].str.len() >= 2) & (pairs["english"].str.len() >= 2)]
    pairs["pair_key"] = pairs["hinglish"].str.casefold() + "\u241f" + pairs["english"].str.casefold()
    pairs = pairs.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"]).reset_index(drop=True)
    order = np.random.default_rng(42).permutation(len(pairs))
    cut = int(round(len(pairs) * 0.80))
    result = {}
    for split, indices in {"train": order[:cut], "test": order[cut:]}.items():
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


def build_tatoeba_splits(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    with zipfile.ZipFile(raw_path) as archive:
        text_files = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not text_files:
            raise ValueError("Tatoeba archive contains no text file.")
        source_file = sorted(text_files, key=lambda name: ("hin" not in name.lower(), len(name)))[0]
        with archive.open(source_file) as handle:
            decoded = io.TextIOWrapper(handle, encoding="utf-8-sig", errors="replace")
            records = []
            for line in decoded:
                parts = next(csv.reader([line.rstrip("\r\n")], delimiter="\t"))
                if len(parts) < 2:
                    continue
                english, hindi = normalize_text(parts[0]), normalize_text(parts[1])
                attribution = normalize_text(parts[2]) if len(parts) >= 3 else ""
                if len(english) < 2 or len(hindi) < 2:
                    continue
                records.append({"english": english, "hindi": hindi, "attribution": attribution})

    pairs = pd.DataFrame(records)
    pairs["pair_key"] = pairs["english"].str.casefold() + "\u241f" + pairs["hindi"].str.casefold()
    pairs = pairs.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"]).reset_index(drop=True)
    order = np.random.default_rng(17).permutation(len(pairs))
    cut = int(round(len(pairs) * 0.80))
    result = {}
    for split, indices in {"train": order[:cut], "test": order[cut:]}.items():
        selected = pairs.iloc[indices]
        rows = []
        for row in selected.itertuples(index=False):
            rows.append({"text": row.hindi, "label": "hindi", "source": "Tatoeba"})
            rows.append({"text": row.english, "label": "english", "source": "Tatoeba"})
        result[split] = pd.DataFrame(rows, columns=["text", "label", "source"])

    # Keep the original attribution field alongside the paired text for CC BY compliance.
    attribution_path = ROOT / "DATASET_ATTRIBUTION_TATOEBA.tsv"
    pairs[["english", "hindi", "attribution"]].to_csv(
        attribution_path, sep="\t", index=False, encoding="utf-8"
    )
    audit = {
        "dataset": TATOEBA_NAME,
        "source_url": TATOEBA_SOURCE_URL,
        "license": TATOEBA_LICENSE,
        "source_file": source_file,
        "clean_unique_pairs": len(pairs),
        "split_rule": "deterministic 80/20 split by paired row; English and Hindi sentence pair stay in same split",
        "attribution_file": "DATASET_ATTRIBUTION_TATOEBA.tsv",
        "label_counts": {
            split: {label: int((df["label"] == label).sum()) for label in LABELS}
            for split, df in result.items()
        },
    }
    return result["train"], result["test"], audit, pairs


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

    tatoeba_path = RAW_DIR / "hin-eng.zip"
    download(TATOEBA_URL, tatoeba_path)
    tatoeba_train, tatoeba_test, tatoeba_audit, _ = build_tatoeba_splits(tatoeba_path)

    frames = {
        "train": clean_combined(pd.concat(
            [comi_frames["train"], phinc_train, tatoeba_train], ignore_index=True
        )),
        "test": clean_combined(pd.concat(
            [comi_frames["test"], phinc_test, tatoeba_test], ignore_index=True
        )),
    }
    test_keys = set(frames["test"]["text"].str.casefold())
    before = len(frames["train"])
    frames["train"] = frames["train"][
        ~frames["train"]["text"].str.casefold().isin(test_keys)
    ].reset_index(drop=True)

    audit = {
        "dataset": "COMI-LINGUA + PHINC + Tatoeba English-Hindi",
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
                "use": "Romanized Hinglish source text and human English translations; paired rows split together",
            },
            {
                "name": TATOEBA_NAME,
                "url": TATOEBA_SOURCE_URL,
                "license": TATOEBA_LICENSE,
                "use": "monolingual Hindi and English examples; paired rows split together; attribution retained in DATASET_ATTRIBUTION_TATOEBA.tsv",
            },
        ],
        "license": "COMI-LINGUA conservatively treated as CC BY-SA 4.0; PHINC CC BY 4.0; Tatoeba CC BY 2.0 France. See DATASET_CARD.md and attribution file.",
        "label_rule": f"COMI per-annotator: hi ratio >= {PURE_LANGUAGE_THRESHOLD:.0%} => hindi; en ratio >= {PURE_LANGUAGE_THRESHOLD:.0%} => english; otherwise hinglish; majority vote across human annotators; PHINC source => hinglish and translation => english; Tatoeba pair => hindi/english",
        "cleaning": [
            "Unicode NFKC normalization",
            "remove zero-width/BOM characters and collapse whitespace",
            "majority vote across human COMI-LINGUA annotators; discard ties",
            "drop empty/very short text",
            "deduplicate exact case-insensitive text within each combined split",
            "split PHINC and Tatoeba by paired row before expanding labels",
            "remove exact train/test text overlap across all sources",
        ],
        "comi_splits": comi_audit,
        "phinc": phinc_audit,
        "tatoeba": tatoeba_audit,
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
