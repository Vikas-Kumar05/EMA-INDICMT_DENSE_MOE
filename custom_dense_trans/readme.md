# System 1: Dense Transformer (`customtransformer`)

Code for the dense system submitted by team **EMA-IND-MT** (IIT Dharwad) to the
**COILD-INDIC-MT** shared task on direct Indic-to-Indic machine translation
(ICON 2026, primary track: official data only, model trained from scratch).

One multilingual encoder-decoder Transformer translates four directions:
Assamese↔Bodo and Bengali↔Odia. The target language is chosen by a tag at the
start of the source sentence: `<2asm>`, `<2brx>`, `<2ben>`, `<2odi>`.

The sparse mixture-of-experts system (System 2) is in the sibling folder.

## Official results

Scores from the organisers on the hidden test set (primary track).

| Direction | chrF++ | BLEU | TER |
|---|---:|---:|---:|
| Assamese → Bodo | 36.26 | 9.57 | 79.03 |
| Bodo → Assamese | 34.63 | 14.27 | 75.77 |
| Bengali → Odia | 43.32 | 18.57 | 65.86 |
| Odia → Bengali | 43.47 | 19.16 | 63.91 |

Scores printed by `compute_metrics.py` are computed on the test files in
`processed/<run>/`, so they will not match this table exactly.

## Model

| | |
|---|---|
| Architecture | pre-norm encoder-decoder Transformer, 6 encoder + 6 decoder layers |
| Size | d_model 512, 8 heads, SwiGLU feed-forward with d_ff 1024, dropout 0.1 |
| Embeddings | one shared embedding matrix, tied to the output projection, sinusoidal positions |
| Tokenizer | joint SentencePiece BPE, 16,000 pieces, byte fallback, digits split |
| Loss | label-smoothed cross-entropy (0.1) |
| Optimiser | AdamW (betas 0.9/0.98, weight decay 0.01), peak lr 5e-4, 2,000 warm-up steps then inverse-square-root decay, gradient clipping 1.0 |
| Batching | batch size 64, length-bucketed, pairs longer than 128 tokens dropped |
| Stopping | validation loss, patience 6 epochs |
| Decoding | beam size 5, length penalty 0.6 |

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: prepares data, trains the tokenizer, trains the model |
| `compute_metrics.py` | Entry point: beam-search decoding and evaluation on the test set |
| `text_utils.py` | Normalisation, noise filters, script check, output post-processing |
| `indic_proc.py` | Wrapper around IndicTransToolkit (pre- and post-processing) |
| `tokeniser.py` | Cleaning, splitting, bidirectional data, SentencePiece training |
| `prepare_dataset.py` | Dataset caching and length-bucketed batching |
| `train.py` | Training loop, early stopping, checkpoints, logs |
| `model.py`, `encoder.py`, `decoder.py`, `embeddings.py` | The model |

## Install

```bash
pip install torch sentencepiece sacrebleu pandas openpyxl numpy scipy tqdm editdistance
pip install indic-nlp-library IndicTransToolkit
```

Check that the toolkit loads:

```bash
python -c "from indic_proc import toolkit_available; print(toolkit_available())"   # True
```

## Data

The shared-task data is not included. Download it from the task page
(<https://coildsharedtask.github.io/home/>) and put it in `data/` (a header row
is required, extra columns are ignored):

```
data/asm_bodo.csv         columns: asm, bodo        (.xlsx also works)
data/bengali_odia.csv     columns: bengali, odia
```

If `asm_bodo_dev.csv`, `asm_bodo_test.csv`, `bengali_odia_dev.csv` and
`bengali_odia_test.csv` are present they are used as the dev and test sets.
Otherwise 1,000 dev and 1,000 test rows are split off the training file
(capped at 5% of the data each).

## Train

```bash
python main.py --indic-mode toolkit --run-name system1
```

This cleans the data, builds the bidirectional training file, trains
SentencePiece, then trains the model. Other useful options:

```bash
python main.py --pairs asm-brx                 # one pair only
python main.py --amp                           # bf16 on GPU
python main.py --resume checkpoints/system1/checkpoint_epoch_10.pt   # same --run-name and --indic-mode
```

Outputs:

```
processed/<run>/      cleaned data, test files, preproc.json
tokenizer/<run>/      spm.model
checkpoints/<run>/    best_model.pt, best_bleu_model.pt, checkpoint_epoch_N.pt, logs/
```

`best_model.pt` is the checkpoint with the lowest validation loss. When early
stopping fires, the last epoch is also saved as `checkpoint_epoch_N_early_stop.pt`.

## Evaluate

```bash
python compute_metrics.py --run-name system1
```

By default this loads `checkpoints/<run>/best_model.pt`. Use `--checkpoint <file>`
to pick another one. Results are written to `evaluation_results/<run>/`:
`evaluation_summary.txt`, `predictions_*.txt`, `comparison_*.txt`,
`error_analysis_*.txt`. The metrics are BLEU, chrF, chrF++, TER, CER, WER,
character F1 and LeBLEU, with a bootstrap confidence interval for BLEU. spBLEU
is added when the FLORES tokenizer can be downloaded.

The IndicTransToolkit mode is read from `processed/<run>/preproc.json`, so
evaluation always uses the same setting as training.

## What the pipeline does

**Cleaning** (training rows only; dev and test are never filtered)
- Unicode NFC, invisible characters removed (ZWJ and ZWNJ are kept), punctuation variants mapped to one form, native digits mapped to ASCII, Indic-NLP normalisation (nukta and vowel-sign order).
- Rows dropped for: wrong script (less than half of the letters in the expected script), URLs, HTML or e-mail, mostly digits, repetitive text, more than 150 words, extreme character or word length ratio, identical source and target, and exact duplicates.
- Training rows whose source or target also appears in dev or test are removed.

**IndicTransToolkit** (`--indic-mode toolkit`)
- Before training: normalise, tokenise, and transliterate Assamese, Bengali and Odia into Devanagari. Bodo is already Devanagari, so all four languages share one script and one vocabulary.
- After decoding: transliterate back to the native script and detokenise.
- References go through the same round trip at evaluation time, so formatting-only differences do not cost score.
- The mode can only be chosen before training. Use a new `--run-name` when changing it.

**Decoding**
- Beam search, one beam at a time, with `<unk>` banned. The maximum output length is twice the source length (between 50 and 128 tokens).

**Post-processing** (on by default)
- `<unk>` symbols and language tags removed.
- Latin-script words dropped unless they also appear in the source (`--latin source-only|keep|drop`).
- Repeated word loops collapsed, and a missing final `।`, `?` or `!` restored from the source.
- Numbers written in the native digits of the target language (`--digits native|ascii`).

## Reproducibility notes

- Seed is 42 (`--seed`). Results can still vary slightly between GPUs and library versions.
- The IndicTransToolkit and `indic-nlp-library` versions affect the pre-processing, so note them if you want exact reproduction.
- Run names are tied to their tokenizer. Changing `--indic-mode` with an existing run name rebuilds the data and overwrites that run's tokenizer.

## Citation

If you use this code, please cite our system description paper:

```bibtex
@inproceedings{emaindmt2026coild,
  title     = {TODO: final paper title},
  author    = {TODO: authors},
  booktitle = {Proceedings of ICON 2026: COILD-INDIC-MT Shared Task},
  year      = {2026}
}
```

## License

TODO: add a license (for example MIT) before making the repository public.
