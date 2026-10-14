# Dataset Card: Hindi-English-Hinglish Language Classifier

## Primary source

- **Dataset:** COMI-LINGUA, curated by the Lingo Research Group at IIT Gandhinagar.
- **Source repository:** https://github.com/lingo-iitgn/COMI-LINGUA
- **Task used:** token-level language identification (LID) annotations, aggregated to sentence labels.
- **Size:** source reports 181,463 annotated instances across its tasks; the exact number retained for this classifier is recorded in `data/processed/dataset_audit.json` during the build.
- **License:** upstream GitHub repository declares CC BY-SA 4.0 in its repository/license badge and the Hugging Face card lists CC BY 4.0. To be conservative, this project records CC BY-SA 4.0. Attribution and share-alike obligations must be respected. Confirm the license with dataset maintainers before redistributing or using in a high-stakes commercial deployment.
- **Paper:** Sheth, Beniwal, Singh (2025), “COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing,” Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.422/

## Label construction

The source provides token-level `hi`, `en`, and other tags. Sentence labels are derived as follows:
- At least one `hi` and one `en` token: `hinglish`
- `hi` tokens and no `en` tokens: `hindi`
- `en` tokens and no `hi` tokens: `english`
- Other-only or unlabelled examples are discarded.

This is a transparent heuristic aggregation, not a source-provided sentence-level label. Inspect class counts in the generated audit before trusting metrics.

## Cleaning and split integrity

The acquisition script applies Unicode NFKC normalization, removes zero-width/BOM characters, collapses whitespace, removes rows with no usable language tags, drops empty/very short examples, deduplicates within each official split, and removes exact train/test text overlap. The official source test split remains held out.

Raw and processed corpus files are not committed to Git; the training workflow downloads them when it runs. The generated audit records row counts, class balance, and cleaning losses.

## Important limitations

COMI-LINGUA is a strong expert-annotated code-mixing resource, but it is not a complete production dataset for spoken voice-agent STT. It does not guarantee coverage of the exact STT engine, regional accents, noisy calls, short acknowledgements, interruptions, or ASR substitutions. A production release must be evaluated on consented, anonymized, speaker-separated transcripts from the target STT pipeline. Never claim production accuracy from this corpus alone.

## Citation

Please cite the COMI-LINGUA paper and follow the source license terms when using or redistributing derived data/models.
