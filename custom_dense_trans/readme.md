
# Transformer MT: Assamese↔Bodo and Bengali↔Odia

One multilingual Transformer (pre-norm, SwiGLU, tied embeddings) that translates in four directions.
The target language is chosen by a tag at the start of the source: `<2asm>`, `<2brx>`, `<2ben>`, `<2odi>`.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: prepares data, trains the tokenizer, trains the model |
| `compute_metrics.py` | Entry point: beam-search evaluation on the test set |
| `text_utils.py` | Normalisation, noise filters, output post-processing, file loading |
| `indic_proc.py` | IndicTransToolkit wrapper (pre- and post-processing) |
| `tokeniser.py` | Cleaning, splitting, bidirectional data, SentencePiece training |
| `prepare_dataset.py`, `train.py` | Dataset/batching and the training loop |
| `model.py`, `encoder.py`, `decoder.py`, `embeddings.py` | The model |

## 1. Install

```bash
pip install torch sentencepiece sacrebleu pandas openpyxl numpy scipy tqdm editdistance
pip install indic-nlp-library IndicTransToolkit      # needed for the Indic preprocessing
```

Check the toolkit works:

```bash
python -c "from indic_proc import toolkit_available; print(toolkit_available())"   # should print True
```

If it prints `False`, the code still runs, but with `--indic-mode off` (see section 5).

## 2. Put the data in place

Create a `data/` folder next to the code (a header row is required, extra columns are ignored):

```
data/asm_bodo.csv         columns: asm, bodo        (.xlsx also works)
data/bengali_odia.csv     columns: bengali, odia
```

Optional ready-made splits, used automatically if present (otherwise 1000 dev and 1000 test rows are split off):
`asm_bodo_dev.csv`, `asm_bodo_test.csv`, `bengali_odia_dev.csv`, `bengali_odia_test.csv`.

## 3. Run, step by step

### Step 1: Train (this also prepares the data and the tokenizer)

```bash
python main.py                       # both pairs, one model, toolkit if installed
```

Useful variants:

```bash
python main.py --pairs asm-brx                       # one pair only
python main.py --pairs ben-odi --epochs 60
python main.py --dropout 0.3                         # recommended if you have < 200k sentences
python main.py --amp                                 # bf16 on GPU, faster
python main.py --asm-bodo-file /path/file.xlsx --ben-odi-file /path/file2.xlsx
```

`main.py` does these things in order: clean and split the data, write the bidirectional train/dev CSVs,
train SentencePiece, then train the Transformer. Outputs:

```
processed/<run>/      cleaned data, test files, preproc.json
tokenizer/<run>/      spm.model
checkpoints/<run>/    best_model.pt, best_bleu_model.pt, logs/
```

`<run>` is `asm-brx+ben-odi` by default, or whatever you pass with `--run-name`.

### Step 2: Evaluate

```bash
python compute_metrics.py --run-name "asm-brx+ben-odi"
```

Uses `checkpoints/<run>/best_model.pt`. Other options: `--checkpoint <file>`, `--beam-size 5`, `--output-dir <dir>`, `--digits native|ascii`, `--latin source-only|keep|drop`, `--pairs`.
The language pairs are read from the run itself, so any `--run-name` works.
Results go to `evaluation_results/<run>/`:
`evaluation_summary.txt` (all metrics per direction), `predictions_*.txt`, `comparison_*.txt`, `error_analysis_*.txt`.

Report the numbers from `compute_metrics.py`. The BLEU printed during training is a quick greedy estimate on 300 dev sentences.

### Step 3: Resume an interrupted run

```bash
python main.py --resume checkpoints/<run>/checkpoint_epoch_10.pt
```

Use the same `--run-name` and the same `--indic-mode` as the original run.

## 4. What the preprocessing does

**Always on (`text_utils.py`)**
- Unicode NFC and removal of invisible characters (ZWJ/ZWNJ are kept because they change conjunct shapes).
- Typographic quotes, dashes, `…` and fullwidth punctuation are mapped to one form; repeated `।।`, `??` are collapsed.
- Bengali, Devanagari and Odia digits become ASCII digits.
- Indic-NLP normaliser (nukta and vowel-sign order), when `indic-nlp-library` is installed.
- Training rows are dropped for: wrong script, URL/HTML/e-mail, mostly digits, repetitive text, over `--max-words` words, extreme length or word-count ratio, identical source and target, exact duplicates.
  Dev and test files are never filtered. Train rows that also appear in dev/test are removed.
- After decoding: repeated loops like "a b a b a b" are collapsed, and a missing final `।`/`?` is restored from the source.

**Output clean-up (`compute_metrics.py`, on by default)**
- **Native digits:** numbers in the output are written in the target language's own digits: `2024` becomes `২০২৪` (Assamese, Bengali), `२०२४` (Bodo), `୨୦୨୪` (Odia). A number glued to a Latin letter (`CO2`, `3D`, `400K`, `0.06eV`) is left alone because it is a formula or unit. Metrics ignore digit style, so this never changes your scores.
  Turn off with `--digits ascii`.
- **Invented English words:** the model sometimes produces fake Latin words. `--latin source-only` (default) keeps a Latin word only if it appears in the source sentence (acronyms, units, formulas, copied terms) and drops the rest. Other values: `--latin keep` (leave everything), `--latin drop` (remove all Latin text). Brackets emptied by this are removed.
- **`⁇` and `�`:** never written to the output. Beam search can no longer pick `<unk>`.
- **Odia ଯ଼ → ୟ:** one spelling of "ya" is used for training data, outputs and references.

**Tokenizer: no more `<unk>` / `⁇`**
- SentencePiece is trained with byte-fallback, so Latin letters, Greek symbols (π, λ) and any other character outside the vocabulary are kept as bytes instead of turning into `⁇`. This uses 256 vocabulary slots, so `--vocab-size` must be at least 600 (16000 is typical).
- **A tokenizer trained with an older version of this code has no byte-fallback**, so it will keep producing `⁇`. Retrain: `main.py` detects the old version and rebuilds data and tokenizer, which makes old checkpoints of that run unusable. Use a new `--run-name` to keep the old run.

**IndicTransToolkit (`--indic-mode toolkit`)**
- Before training: normalise, tokenise, and transliterate every language into Devanagari, so related languages share one script and one vocabulary.
- After decoding: transliterate back to the native script and detokenise.
- At evaluation: references go through the same round trip, so formatting-only differences (digits, ellipsis, spacing) cost no BLEU.

## 5. When can I turn the IndicTransToolkit on or off?

**Only before training starts, not during a run.** The mode changes the text the tokenizer and the model see
(native script vs Devanagari), so the vocabulary and all weights depend on it. Switching halfway would make the
checkpoint meaningless.

| When | What to do |
|---|---|
| **Before the first training run** | Choose with `--indic-mode toolkit`, `--indic-mode off`, or the default `auto` (toolkit if it is installed and working, otherwise off). |
| **Compare both on your data** | Train two runs with different names (below). Compare their BLEU/chrF++. |
| **After training, at evaluation** | Nothing to choose. `compute_metrics.py` reads `processed/<run>/preproc.json` and uses the same mode as training. |
| **Mid-run (resume)** | Not possible. Resume with the same mode. |

Comparing the two settings:

```bash
python main.py --indic-mode toolkit --run-name with_toolkit
python main.py --indic-mode off     --run-name without_toolkit

python compute_metrics.py --run-name with_toolkit
python compute_metrics.py --run-name without_toolkit
```

Always give each mode its own `--run-name`. If you change `--indic-mode` while re-using a run name, `main.py` rebuilds the
data and **overwrites that run's tokenizer**, and the old checkpoints in that run no longer match it.

## 6. Quick check before a long run

A 2-epoch run on a tiny model, to confirm everything works end to end:

```bash
python main.py --epochs 2 --d-model 64 --nhead 4 --d-ff 128 --enc-layers 1 --dec-layers 1 \
               --vocab-size 800 --dev-size 30 --test-size 30 --num-workers 0 --run-name smoke
python compute_metrics.py --run-name smoke
```

The smoke test needs a corpus of at least a few hundred sentences. Too small a corpus for the chosen `--vocab-size` makes SentencePiece abort.

## 7. Main options

| Option | Default | Meaning |
|---|---|---|
| `--pairs` | both | `asm-brx`, `ben-odi` |
| `--indic-mode` | `auto` | `toolkit`, `off`, `auto` |
| `--dedupe-sides` | off | keep one row per distinct source and per distinct target sentence |
| `--max-words` | 150 | drop training rows longer than this |
| `--vocab-size` | 16000 | SentencePiece vocabulary (minimum 600) |
| `--max-len` | 128 | max tokens per training sentence (longer pairs are dropped) |
| `--epochs` | 50 | upper limit; early stopping uses `--patience` (6) |
| `--batch-size`, `--lr` | 64, 5e-4 | |
| `--dropout` | 0.1 | use 0.3 for small datasets |
| `--force-reprocess` | off | rebuild data and tokenizer |

## 8. Troubleshooting

- **`IndicTransToolkit not usable`**: install `IndicTransToolkit` and `indic-nlp-library`. The run continues with the toolkit off.
- **`ImportError` about `PreTrainedTokenizerBase`** when importing the toolkit directly: caused by the installed `transformers` version. `indic_proc.py` works around it, so you can ignore it.
- **Outputs still contain `⁇`**: the run was trained with an older tokenizer. Retrain (see section 4).
- **SentencePiece `Check failed: freq > 0`**: the corpus is too small for the `--vocab-size`. Lower it (not below 600) or use more data.
- **`spBLEU skipped`**: needs a one-off download of the FLORES tokenizer, so it needs internet access. The other metrics are unaffected.
- **Evaluation says `preproc.json not found`**: the run was prepared with an older version of the code. Re-run `main.py --force-reprocess`.
- **`CUDA out of memory`**: lower `--batch-size` and raise `--grad-accum-steps`.
