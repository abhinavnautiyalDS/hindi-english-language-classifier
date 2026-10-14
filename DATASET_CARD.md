# Dataset Card: Hindi-English-Hinglish Language Classifier

## Sources

### 1. COMI-LINGUA

- **Curator:** Lingo Research Group, IIT Gandhinagar.
- **Source:** https://github.com/lingo-iitgn/COMI-LINGUA
- **Use:** official LID train/test splits; language tags from the human annotator columns are preferred over the model-predicted tag column.
- **License:** the upstream GitHub repository declares CC BY-SA 4.0, while a Hugging Face copy's metadata has stated CC BY 4.0. This project conservatively treats the source as CC BY-SA 4.0. Confirm with the maintainers if needed.
- **Paper:** Sheth, Beniwal, Singh (2025), “COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing,” Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.422/

### 2. PHINC

- **Curator:** Lingo Research Group, IIT Gandhinagar.
- **Source:** https://huggingface.co/datasets/LingoIITGN/PHINC
- **Use:** Romanized Hinglish source text and human-generated English translations. Each pair is assigned to a single split before being expanded into labeled examples.
- **Size:** the source reports 13,738 Hinglish-English pairs.
- **License:** CC BY 4.0.
- **Paper:** Srivastava and Singh (2020), “PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation,” W-NUT 2020. https://aclanthology.org/2020.wnut-1.7/

### 3. Tatoeba English-Hindi

- **Source:** https://mail.manythings.org/anki/hin-eng.zip
- **Selection/description:** English-Hindi sentence pairs selected from Tatoeba and distributed by ManyThings.org. The source describes a checked selection of pairs and includes sentence-level attribution metadata.
- **Use:** additional monolingual Hindi and English text. English and Hindi from each pair are kept in the same train/test partition.
- **License:** CC BY 2.0 France. The pipeline preserves source attribution metadata in `DATASET_ATTRIBUTION_TATOEBA.tsv`.

## Label construction

The final three labels are project-defined sentence-level labels:

- **COMI-LINGUA:** for each annotator, Hindi token ratio >= 80% maps to `hindi`, English token ratio >= 80% maps to `english`, and otherwise maps to `hinglish`. A majority vote across human annotators determines the sentence label; ties or rows without usable tags are discarded.
- **PHINC:** the Hinglish source sentence maps to `hinglish`; its human English translation maps to `english`.
- **Tatoeba:** the Hindi sentence maps to `hindi`; the English sentence maps to `english`.

These labels are not all natively provided by the source datasets. The exact rule, per-source row counts, cleaning counts, and final class counts are recorded in `data/processed/dataset_audit.json` during each build.

## Cleaning and split integrity

The pipeline applies Unicode NFKC normalization, removes zero-width/BOM characters, collapses whitespace, drops empty/very short examples, deduplicates case-insensitive exact text within splits, splits PHINC and Tatoeba by translation pair, and removes exact text overlap between the combined train/test sets. The COMI-LINGUA held-out split is retained.

The cleaned, labeled `data/processed/train.csv` and `data/processed/test.csv` datasets, plus `data/processed/dataset_audit.json`, are committed to the repository for inspection and reproducibility. Raw source downloads remain excluded from Git and are downloaded by the build pipeline. The Tatoeba attribution file is retained in the repository to preserve source attribution.

## Important limitations

This is a stronger text-corpus baseline, not a finished production model. PHINC is social-media code-mixed text; Tatoeba is sentence-pair text; COMI-LINGUA is a code-mixing corpus. None guarantees coverage of real spoken calls, regional accents, interruptions, short acknowledgements, noisy audio, or recognition errors from your exact STT engine.

Before production, collect consented and anonymized transcripts from the target STT system, separate speakers/calls across train and test, review labels with native speakers, and report metrics on that untouched evaluation set. Do not claim production accuracy from public corpus metrics alone.

## Licensing note

The sources have different licenses. Follow the applicable attribution and share-alike terms before redistributing the resulting model or using it in a commercial deployment. The model's provenance and licenses are recorded in `models/evaluation.json`.
