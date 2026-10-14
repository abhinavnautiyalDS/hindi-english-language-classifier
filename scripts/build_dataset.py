from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS = ("hindi", "english", "hinglish")

# COMI-LINGUA: expert-reviewed Hindi-English code-mixing annotations.
# Use the upstream CSVs and retain their official train/test split.
SOURCE_FILES = {
    "train": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/LID_train.csv?download=true",
    "test": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/LID_test.csv?download=true",
}
SOURCE_NAME = "COMI-LINGUA (Lingo Research Group, IIT Gandhinagar)"
SOURCE_LICENSE = "CC BY-SA 4.0 (conservative interpretation; upstream GitHub repository declares CC-BY-SA-4.0)"
SOURCE_URL = "https://github.com/lingo-iitgn/COMI-LINGUA"


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
    text = re.sub(r"\s+", " ", text).strip()
    return text


def choose_columns(frame: pd.DataFrame) -> tuple[str, str]:
    normalized = {col: re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in frame.columns}
    text_candidates = [
        col for col, name in normalized.items()
        if any(key in name for key in ("sentence", "utterance", "text", "transcript"))
    ]
    tag_candidates = [
        col for col, name in normalized.items()
        if any(key in name for key in ("lid", "languagetag", "languagelabel", "tags", "label"))
        and col not in text_candidates
    ]
    if not text_candidates or not tag_candidates:
        raise ValueError(
            "Could not identify sentence and token-language-tag columns. "
            f"Found columns: {list(frame.columns)}"
        )
    # Prefer the most explicit header names when there are several candidates.
    text_col = sorted(text_candidates, key=lambda c: (("sentence" not in normalized[c]), len(normalized[c])))[0]
    tag_col = sorted(tag_candidates, key=lambda c: (("lid" not in normalized[c] and "tag" not in normalized[c]), len(normalized[c])))[0]
    return text_col, tag_col


def sentence_label(tags_value: object) -> str | None:
    # LID tags are token-level values such as "hi hi en en ot".
    # Ignore punctuation/other tokens; classify mixed sentences as Hinglish.
    tags = re.findall(r"(?<![a-z])(hi|en|ot|other)(?![a-z])", str(tags_value).lower())
    has_hi = "hi" in tags
    has_en = "en" in tags
    if has_hi and has_en:
        return "hinglish"
    if has_hi:
        return "hindi"
    if has_en:
        return "english"
    return None


def convert_split(split: str, raw_path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(raw_path, low_memory=False)
    text_col, tag_col = choose_columns(frame)
    print(f"{split}: source columns = {list(frame.columns)}")
    data = pd.DataFrame({
        "text": frame[text_col].map(normalize_text),
        "label": frame[tag_col].map(sentence_label),
    })
    before = len(data)
    data = data.dropna(subset=["label"])
    data = data[data["text"].str.len() >= 2]
    after_label_filter = len(data)
    data["text_key"] = data["text"].str.casefold()
    data = data.drop_duplicates(subset=["text_key"], keep="first")
    data = data[["text", "label"]].reset_index(drop=True)
    counts = {label: int((data["label"] == label).sum()) for label in LABELS}
    audit = {
        "source_split": split,
        "input_rows": before,
        "rows_with_usable_language_tags": after_label_filter,
        "rows_after_exact_casefold_deduplication": len(data),
        "dropped_rows": before - len(data),
        "label_counts": counts,
        "text_column": text_col,
        "tag_column": tag_col,
    }
    return data, audit


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    audit = {
        "dataset": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "license": SOURCE_LICENSE,
        "label_rule": "hi+en tags => hinglish; hi only => hindi; en only => english; ot/other ignored",
        "cleaning": [
            "Unicode NFKC normalization",
            "remove zero-width/BOM characters",
            "collapse whitespace",
            "drop rows without usable hi/en language tags",
            "drop empty/very short texts",
            "case-insensitive exact deduplication within each official split",
            "remove training rows that overlap the official test set",
        ],
        "splits": {},
    }

    for split, url in SOURCE_FILES.items():
        raw_path = RAW_DIR / f"COMI-LINGUA_LID_{split}.csv"
        download(url, raw_path)
        frames[split], audit["splits"][split] = convert_split(split, raw_path)

    # Keep official test examples completely out of training, even if upstream duplicates exist.
    test_keys = set(frames["test"]["text"].str.casefold())
    before = len(frames["train"])
    frames["train"] = frames["train"][
        ~frames["train"]["text"].str.casefold().isin(test_keys)
    ].reset_index(drop=True)
    audit["train_test_overlap_removed"] = before - len(frames["train"])
    audit["final_label_counts"] = {
        split: {label: int((df["label"] == label).sum()) for label in LABELS}
        for split, df in frames.items()
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
    print(json.dumps(audit["final_label_counts"], indent=2))
    print(f"Clean train/test files written to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
