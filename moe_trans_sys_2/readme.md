
# System 2: Sparse Mixture-of-Experts Transformer (`custommoetransformer`)

Code for the sparse system submitted by team **EMA-IND-MT** (IIT Dharwad) to the
**COILD-INDIC-MT** shared task on direct Indic-to-Indic machine translation
(ICON 2026, primary track: official data only, model trained from scratch).

One multilingual encoder-decoder Transformer translates four directions:
Assamese↔Bodo and Bengali↔Odia. The target language is chosen by a tag at the
start of the source sentence: `<2asm>`, `<2brx>`, `<2ben>`, `<2odi>`.

This is the dense Transformer of System 1 with the feed-forward block of every
second layer replaced by a sparse mixture of experts. The dense system is in the
sibling folder.

## Official results

Scores from the organisers on the hidden test set (primary track).

| Direction | chrF++ | BLEU | TER |
|---|---:|---:|---:|
| Assamese → Bodo | 34.86 | 9.68 | 79.78 |
| Bodo → Assamese | 31.20 | 12.00 | 79.13 |
| Bengali → Odia | 42.39 | 17.86 | 67.24 |
| Odia → Bengali | 39.87 | 16.10 | 67.56 |

Scores printed by `compute_metrics.py` are computed on the test files in
`processed/<data_name>/`, so they will not match this table exactly.

## Model

| | |
|---|---|
| Backbone | pre-norm encoder-decoder Transformer, 6 encoder + 6 decoder layers |
| Size | d_model 512, 8 heads, SwiGLU feed-forward with d_ff 1024, dropout 0.1 |
| Embeddings | one shared embedding matrix, tied to the output projection, sinusoidal positions |
| MoE layers | every 2nd layer (2, 4, 6) of the encoder and of the decoder |
| Experts | 4 SwiGLU experts per MoE layer, each the same size as the dense feed-forward block |
| Routing | linear router, top-2 experts per token, outputs mixed with the renormalised router probabilities |
| Capacity | no capacity limit, so no token is dropped; padding tokens are never routed |
| Auxiliary losses | load-balancing loss (weight 0.01) and router z-loss (weight 0.001) |
| Tokenizer | joint SentencePiece BPE, 16,000 pieces, byte fallback, digits split |
| Loss | label-smoothed cross-entropy (0.1) plus the two auxiliary losses |
| Optimiser | AdamW (betas 0.9/0.98, weight decay 0.01), peak lr 5e-4, 2,000 warm-up steps then inverse-square-root decay, gradient clipping 1.0 |
| Batching | batch size 64, length-bucketed, pairs longer than 128 tokens dropped |
| Stopping | validation loss, patience 6 epochs |
| Decoding | batched beam search, beam size 5, length penalty 0.6 |

The validation loss is plain cross-entropy without the auxiliary terms, so it can be
compared with the dense model.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: prepares data, trains the tokenizer, trains the model |
| `compute_metrics.py` | Entry point: beam-search decoding and evaluation on the test set |
| `translate.py` | Entry point: translate a sentence or a text file with a trained run |
| `moe.py` | SwiGLU block and the sparse top-k mixture-of-experts layer |
| `model.py`, `encoder.py`, `decoder.py`, `embeddings.py` | The model |
| `inference.py` | Model loading, batched beam search, full translation pipeline |
| `postprocess.py` | Output clean-up: native digits, Latin-word filter, nukta recomposition |
| `indic_utils.py` | Wrapper around IndicTransToolkit |
| `text_utils.py` | Language registry, normalisation, script check, file loading |
| `tokeniser.py` | Cleaning, splitting, bidirectional data, SentencePiece training |
| `prepare_dataset.py`, `train.py` | Dataset caching, batching, and the training loop |

## Install

```bash
pip install torch sentencepiece sacrebleu pandas openpyxl numpy scipy tqdm editdistance
pip install IndicTransToolkit
```

IndicTransToolkit needs a C compiler to build. If `import IndicTransToolkit` fails
because of the `transformers` version, `indic_utils.py` falls back to loading the
compiled processor directly.

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
python main.py --indictrans --moe --run-name indic_moe
```

This cleans the data, applies the IndicTrans pre-processing, builds the
bidirectional training file, trains SentencePiece, then trains the model.
Other useful options:

```bash
python main.py --indictrans --moe --amp                 # bf16 on GPU
python main.py --indictrans --moe --moe-experts 8 --moe-top-k 2
python main.py --indictrans --moe --resume checkpoints/indic_moe/checkpoint_epoch_10.pt
python main.py --run-name baseline                      # same code, dense, no toolkit
```

Without `--moe` the same code trains the dense model, and without `--indictrans` it
skips the toolkit.

Outputs:

```
processed/<data_name>/     cleaned data, test files, meta.json
tokenizer/<data_name>.bf/  spm.model
checkpoints/<run>/         best_model.pt, checkpoint_epoch_N.pt, logs/
```

`best_model.pt` is the checkpoint with the lowest validation loss. When early
stopping fires, the last epoch is also saved as `checkpoint_epoch_N_early_stop.pt`.
`logs/training.log` records the load-balancing loss and, for every MoE layer, the
share of tokens sent to each expert after each epoch.

## Evaluate

```bash
python compute_metrics.py --run-name indic_moe
```

By default this loads `checkpoints/<run>/best_model.pt`. Use `--checkpoint <file>`
to pick another one. The data folder, tokenizer, language pairs, IndicTrans setting
and MoE layout are all read from the checkpoint, so a run is evaluated the way it was
trained. Results are written to `evaluation_results/<run>/`:
`evaluation_summary.txt`, `results.json`, `predictions_*.txt`, `comparison_*.txt`,
`error_analysis_*.txt`. The metrics are BLEU, chrF, chrF++, TER, CER, WER,
character F1 and LeBLEU, with a bootstrap confidence interval for BLEU.

## Translate

```bash
python translate.py --run-name indic_moe --src asm --tgt brx --text "..."
python translate.py --run-name indic_moe --src ben --tgt odi --input in.txt --output out.txt
```

Language codes are `asm`, `brx`, `ben` and `odi`. Valid directions are asm↔brx and
ben↔odi. The input file has one sentence per line.

## What the pipeline does

**Cleaning** (applied to training, dev and test files)
- Unicode NFC, invisible characters removed (ZWJ and ZWNJ are kept), control characters turned into spaces.
- Rows dropped for: wrong script (less than half of the letters in the expected script), character length ratio above 3.5, identical source and target, and exact duplicates. The length-ratio limit is switched off for dev and test files.
- Training rows whose source or target also appears in dev or test are removed.

**IndicTransToolkit** (`--indictrans`)
- Before training: normalise punctuation, map native digits to ASCII, tokenise, and transliterate Assamese, Bengali and Odia into Devanagari. Bodo is already Devanagari, so all four languages share one script and one vocabulary.
- After decoding: transliterate back to the native script and detokenise.
- Placeholders for URLs and numbers are off by default (`--indic-placeholders`).

**Tokenizer**
- Latin letters, digits, ASCII punctuation, Greek letters and common symbols are always in the vocabulary, and byte fallback covers anything else.

**Decoding**
- All beams are decoded in one batched step. Finished hypotheses leave the beam and are ranked by length-normalised score. The maximum output length is twice the source length (between 50 and 128 tokens).
- If the source contains characters the tokenizer cannot represent, `<unk>` is allowed and each one is replaced by the next such source character. Otherwise `<unk>` is banned. Language tags can never be generated.

**Post-processing** (on by default)
- Artefacts (`<unk>` symbols, tags) removed.
- Latin-script words dropped unless they also appear in the source (`--latin-policy source|keep|drop`).
- Numbers written in the native digits of the target language (`--digits native|ascii|keep`).
- Nukta characters recomposed for Bengali, Assamese and Odia (`--no-nukta-recompose` turns this off).
- Scoring uses a canonical form (digits as ASCII, straight quotes), so the digit style does not change BLEU or chrF.

## Differences from System 1

The two systems share the Transformer backbone, but they come from separate code
bases, so the comparison is not a controlled ablation of the experts alone.

| | System 1 (dense) | System 2 (MoE) |
|---|---|---|
| Feed-forward | dense SwiGLU in every layer | 4 experts, top-2, every 2nd layer |
| Cleaning | adds URL/HTML, digit-heavy, repetitive and word-ratio filters, punctuation mapping and Indic-NLP normalisation | script check, length ratio, identical and duplicate rows |
| Beam search | one beam at a time, final pick by raw log-probability | batched, ranked by length-normalised score, `<unk>` copy-back |
| Extra post-processing | repeated-loop collapse, final punctuation restored | nukta recomposition |

## Reproducibility notes

- Seed is 42 (`--seed`). Results can still vary slightly between GPUs and library versions.
- The IndicTransToolkit version affects the pre-processing, so note it if you want exact reproduction.
- Data, tokenizer and cache folders are named after the text processing (`asm-brx+ben-odi.indictrans`), so a dense and an MoE run on the same text share one processed set.


