# EMA-INDICMT_DENSE_MOE

Code for the two systems submitted by team **EMA-IND-MT** (IIT Dharwad) to the
**COILD-INDIC-MT** shared task on direct Indic-to-Indic machine translation
(ICON 2026), primary track: official data only, models trained from scratch.

Both systems are multilingual encoder-decoder Transformers that translate four
directions with one set of weights: Assamese↔Bodo and Bengali↔Odia. The target
language is chosen by a tag at the start of the source sentence
(`<2asm>`, `<2brx>`, `<2ben>`, `<2odi>`).

| System | Folder | Description |
|---|---|---|
| System 1 (`customtransformer`) | [`custom_dense_trans`](custom_dense_trans) | Dense pre-norm Transformer with SwiGLU feed-forward blocks |
| System 2 (`custommoetransformer`) | [`moe_trans_sys_2`](moe_trans_sys_2) | The same backbone with a sparse mixture of experts in every 2nd layer (4 experts, top-2) |

Each folder has its own README with install, training, evaluation and
pre-processing details.

## Official results

Scores from the organisers on the hidden test set (primary track).
chrF++ is the ranking metric. Higher is better for chrF++ and BLEU, lower for TER.

| Direction | System 1 chrF++ | System 1 BLEU | System 1 TER | System 2 chrF++ | System 2 BLEU | System 2 TER |
|---|---:|---:|---:|---:|---:|---:|
| Assamese → Bodo | 36.26 | 9.57 | 79.03 | 34.86 | 9.68 | 79.78 |
| Bodo → Assamese | 34.63 | 14.27 | 75.77 | 31.20 | 12.00 | 79.13 |
| Bengali → Odia | 43.32 | 18.57 | 65.86 | 42.39 | 17.86 | 67.24 |
| Odia → Bengali | 43.47 | 19.16 | 63.91 | 39.87 | 16.10 | 67.56 |
| **Average** | **39.42** | **15.39** | **71.14** | **37.08** | **13.91** | **73.43** |

## Pipeline

Both systems follow the same overall steps:

1. Clean the official parallel data and build a bidirectional training set with target-language tags.
2. Pre-process with the IndicTransToolkit, which transliterates Assamese, Bengali and Odia into Devanagari so that all four languages share one script.
3. Train one joint SentencePiece BPE vocabulary (16,000 pieces, byte fallback).
4. Train the Transformer with label-smoothed cross-entropy and AdamW, with early stopping on validation loss.
5. Decode with beam search (beam 5, length penalty 0.6), transliterate back to the native script, and clean the output.

## What differs between the two systems

The systems share the backbone but come from separate code bases, so the gap
between them is not a controlled ablation of the experts alone.

| | System 1 (dense) | System 2 (MoE) |
|---|---|---|
| Feed-forward | dense SwiGLU in every layer | 4 experts, top-2, every 2nd layer, with load-balancing and router z-losses |
| Cleaning | extra URL/HTML, digit-heavy, repetitive and word-ratio filters, Indic-NLP normalisation | script check, length ratio, identical and duplicate rows |
| Beam search | one beam at a time, final pick by raw log-probability | batched, ranked by length-normalised score, `<unk>` copy-back |
| Extra post-processing | repeated-loop collapse, final punctuation restored | nukta recomposition |

## Repository layout

```
EMA-INDICMT_DENSE_MOE/
├── README.md                 this file
├── custom_dense_trans/       System 1: dense Transformer
└── moe_trans_sys_2/          System 2: sparse MoE Transformer
```

## Data

The shared-task data is not included. Download it from the task page
(<https://coildsharedtask.github.io/home/>) and follow the data instructions in
each folder's README.

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
