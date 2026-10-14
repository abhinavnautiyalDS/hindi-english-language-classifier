# Dataset Card: Hindi-English-Hinglish Language Classifier

## Primary source

- **Dataset:** COMI-LINGUA, curated by the Lingo Research Group at IIT Gandhinagar.
- **Source repository:** https://github.com/lingo-iitgn/COMI-LINGUA
- **Task used:** token-level language identification (LID) annotations, aggregated to sentence labels.
- **Size:** the source reports 181,463 annotated instances across tasks. Exact retained rows for this classifier are recorded in `data/processed/dataset_audit.json` after acquisition.
- **License:** the upstream GitHub repository shows CC BY-SA 4.0, while the Hugging Face dataset card says CC BY 4.0. This project conservatively records CC BY-SA 4.0. Attribution and share-alike obligations must be respected; confirm the intended license with dataset maintainers before redistributing a derived model.
- **Paper:** Sheth, Beniwal, Singh (2025), “COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing,” Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.422/

## Label construction

The source provides token-level `hi`, `en`, and other tags. Sentence labels are derived using the share of Hindi and English tokens:
- Hindi token ratio >= 80%: `hindi`
- English token ratio >= 80%: `english`
- Otherwise: `hinglish`
- Rows without any `hi` or `en` tokens are discarded.

This is a project-defined sentence-level aggregation rule, not a sentence label provided by the source. Review the generated audit and class counts before interpreting metrics.

## Cleaning and split integrity

The acquisition script applies Unicode NFKC normalization, removes zero-width/BOM characters, collapses whitespace, removes rows without usable language tags, drops empty/very short examples, deduplicates within each official split, and removes exact train/test text overlap. The official source test split remains held out.

Raw and processed corpus files are not committed to Git; the training workflow downloads them when it runs. The generated audit records row counts, class balance, and cleaning losses.

## Important limitations

COMI-LINGUA is a valuable expert-annotated code-mixing resource, but it is not a complete production dataset for spoken voice-agent STT. It does not guarantee coverage of the exact STT engine, regional accents, noisy calls, short acknowledgements, interruptions, or ASR substitutions. A production release must be evaluated on consented, anonymized, speaker-separated transcripts from the target STT pipeline. Never claim production accuracy from this corpus alone.

## Citation

Please cite the COMI-LINGUA paper and follow the source license terms when using or redistributing derived data/models.
