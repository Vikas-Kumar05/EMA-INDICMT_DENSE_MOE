"""
compute_metrics.py - Evaluation for Transformer MT Model

METRICS
=======
Standard  : BLEU (13a + no-tok), chrF, chrF++, TER, METEOR
Low-resource: CER, WER, Character F1, LeBLEU
Analysis  : Length stats, Bootstrap confidence intervals,
            Sentence-level BLEU, Worst-translation dump

USAGE
=====
Edit EvalConfig below (paths, beam settings, which metrics to run),
then:  python compute_metrics.py
"""

import torch
import sacrebleu
import sentencepiece as spm
import os
import re
import unicodedata
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import evaluate
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False
    print("⚠️  'evaluate' not found — METEOR / BERTScore will be skipped.")

try:
    import editdistance
    HAS_EDITDISTANCE = True
except ImportError:
    HAS_EDITDISTANCE = False
    print("⚠️  'editdistance' not found — CER / WER will be skipped.")
    print("   Install: pip install editdistance")

# Import from the clean Transformer codebase
from model import TransformerModel
from text_utils import (LANGS, PAIRS, strip_tags, normalize_text, postprocess_output,
                        strip_unk, localize_digits, LATIN_POLICIES)
from indic_proc import IndicText
import json


# ============================================================================
# CONFIGURATION  ← Edit this block before running
# ============================================================================

class EvalConfig:
    # Set from the command line in main():  --run-name / --pairs / --checkpoint
    run_name        = "asm-brx+ben-odi"
    pairs           = ["asm-brx", "ben-odi"]
    # checkpoint_path = None     # default: checkpoints/<run_name>/best_model.pt
    checkpoint_path = "/home/vikas1/vikas/vikas/icon_shared_ind2ind/scratch/v3/checkpoints/asm-brx+ben-odi/checkpoint_epoch_16_early_stop.pt"
    tokenizer_path  = None     # default: tokenizer/<run_name>/spm.model
    test_dir        = None     # default: processed/<run_name>   (<pair>.test.<lang>.txt files)

    # Output
    output_dir = "evaluation_results"

    # Beam search
    beam_size          = 5
    length_ratio       = 2.0      # dynamic max_len = src_len * length_ratio
    base_max_len       = 200
    repetition_penalty = 1.0      # 1.0 = off. 1.5 punishes legitimate repeats (danda, suffixes) in Indic text
    length_penalty     = 0.6
    temperature        = 1.0
    min_decode_len     = 3
    early_stopping     = True

    # Output post-processing
    digits       = "native"       # "native": 2024 -> ২০২৪ / २०२४ / ୨୦୨୪ in the target script; "ascii": leave as is
    latin_policy = "source-only"  # "source-only": drop invented English words, keep those copied from the source
                                  # "keep": leave all Latin text   "drop": remove all Latin text

    # Metrics to compute
    compute_meteor              = False   # English-only (WordNet): meaningless for these languages
    compute_bertscore           = False   # slow; XLM-R covers asm/ben/odi but not Bodo
    compute_cer                 = True
    compute_wer                 = True
    compute_char_f1             = True
    compute_lebleu              = True
    compute_confidence_intervals = True
    num_bootstrap_samples       = 1000

    # Error analysis
    save_sentence_level_scores = True
    save_worst_translations    = True
    num_worst_examples         = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path: str, device: str) -> TransformerModel:
    """Load TransformerModel from a checkpoint saved by train.py."""
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model hyper-params from the saved config dict
    cfg = checkpoint.get('model_config') or checkpoint.get('config', {})
    if 'vocab_size' not in cfg:
        raise ValueError("Checkpoint has no model_config - it was saved by the old train.py. "
                         "Re-train with the updated code.")
    model = TransformerModel(
        vocab_size          = cfg.get('vocab_size', 22000),
        d_model             = cfg.get('d_model', 512),
        nhead               = cfg.get('nhead', 8),
        d_ff                = cfg.get('d_ff', 1024),
        num_encoder_layers  = cfg.get('num_encoder_layers', 6),
        num_decoder_layers  = cfg.get('num_decoder_layers', 6),
        dropout             = cfg.get('dropout', 0.1),
        max_seq_len         = cfg.get('max_seq_len', 128),
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    epoch     = checkpoint.get('epoch', 'unknown')
    best_bleu = checkpoint.get('best_bleu', 'unknown')
    print(f"  ✅ Loaded from epoch {epoch}  (best BLEU: {best_bleu})")
    print(f"  Parameters: {model.get_num_params():,}")

    return model


# ============================================================================
# MASK HELPERS  (self-contained, no dependency on train.py)
# ============================================================================

def _padding_mask(seq, pad_id=0):
    """[B, L] → [B, 1, 1, L]  True = valid token"""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def _causal_mask(size, device):
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


# ============================================================================
# BEAM SEARCH DECODER
# ============================================================================

class BeamSearchDecoder:
    """Beam search decoder for the pure Transformer model."""

    def __init__(self, model: TransformerModel, tokenizer, config: EvalConfig, indic=None):
        self.indic     = indic or IndicText('off')
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config
        self.device    = config.device

        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()
        self.pad_id = tokenizer.pad_id()
        self.unk_id = tokenizer.unk_id()
        self.max_pos = model.embedding.pos_encoder.pe.size(1)   # positional table length

    @torch.no_grad()
    def decode(self, src_text: str, src_lang: str, tgt_lang: str,
               beam_size: int = None, max_len: int = None) -> str:
        """src_text is the RAW native-script sentence (no <2xx> tag): the tag, the
        normalisation and the optional IndicTransToolkit step are applied here exactly
        as they were applied to the training data."""
        beam_size = beam_size or self.config.beam_size

        # Normalise -> (toolkit) -> add target-language tag -> tokenise
        clean_src = normalize_text(src_text, src_lang)
        model_src = self.indic.to_model([clean_src], src_lang)[0]
        src_ids = self.tokenizer.encode(f"{LANGS[tgt_lang]['tag']} {model_src}",
                                        add_bos=True, add_eos=True)
        if len(src_ids) > self.max_pos - 2:                    # never exceed the positional table
            src_ids = src_ids[:self.max_pos - 3] + [self.eos_id]
        src_len = len(src_ids)

        if max_len is None:
            max_len = max(50, min(int(src_len * self.config.length_ratio), 128))
        max_len = min(max_len, self.max_pos - 2)

        min_len = max(self.config.min_decode_len, int(src_len * 0.3))

        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_mask   = _padding_mask(src_tensor, self.pad_id)

        encoder_output = self.model.encode(src_tensor, src_mask)

        if torch.isnan(encoder_output).any():
            return ""

        # Each beam: (token_list, cumulative_log_prob)
        beams     = [([self.bos_id], 0.0)]
        completed = []

        for _step in range(max_len):
            candidates = []

            for tokens, score in beams:
                # Already finished
                if tokens[-1] == self.eos_id:
                    completed.append((tokens, score))
                    continue

                # Hard length cap
                if len(tokens) > src_len * 3:
                    completed.append((tokens + [self.eos_id], score))
                    continue

                tgt_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                T = tgt_tensor.size(1)

                pad_mask = _padding_mask(tgt_tensor, self.pad_id).expand(-1, -1, T, -1)
                causal   = _causal_mask(T, self.device).unsqueeze(0).unsqueeze(0)
                tgt_mask = pad_mask & causal

                try:
                    logits     = self.model.decode_step(tgt_tensor, encoder_output, src_mask)
                    next_logit = logits[0, -1, :].clone()
                except Exception:
                    break

                if torch.isnan(next_logit).any():
                    break

                # Apply temperature
                next_logit = next_logit / self.config.temperature

                # Repetition penalty
                for tok in (set(tokens) if self.config.repetition_penalty != 1.0 else ()):
                    if tok not in (self.bos_id, self.eos_id):
                        if next_logit[tok] > 0:
                            next_logit[tok] /= self.config.repetition_penalty
                        else:
                            next_logit[tok] *= self.config.repetition_penalty

                # Never generate <unk> (it prints as the stray "⁇")
                next_logit[self.unk_id] = float('-inf')

                # Minimum length constraint
                if len(tokens) < min_len:
                    next_logit[self.eos_id] = float('-inf')

                # Nudge EOS for very long outputs
                if len(tokens) > src_len * 2.5:
                    next_logit[self.eos_id] += 2.0

                log_probs = torch.log_softmax(next_logit, dim=-1)

                lp_alpha = 0.4 if src_len > 100 else self.config.length_penalty
                top_lp, top_ids = torch.topk(log_probs, beam_size)

                for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                    new_tokens = tokens + [tok]
                    lp_norm    = ((5 + len(new_tokens)) / 6) ** lp_alpha
                    new_score  = score + lp
                    candidates.append((new_tokens, new_score / lp_norm, new_score))

            if not candidates:
                break

            # Keep top beam_size by normalised score
            ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
            beams   = [(toks, raw) for toks, _norm, raw in ordered[:beam_size]]

            # Early stopping
            if self.config.early_stopping and len(completed) >= beam_size:
                best_done   = max(completed, key=lambda x: x[1])[1]
                best_active = max(beams,     key=lambda x: x[1])[1] if beams else float('-inf')
                threshold   = 0.5 if src_len > 100 else 0.0
                if best_done >= best_active + threshold:
                    break

        # Flush remaining active beams
        for tokens, score in beams:
            if tokens[-1] != self.eos_id:
                completed.append((tokens + [self.eos_id], score))

        best_tokens = max(completed, key=lambda x: x[1])[0] if completed else \
                      (beams[0][0] if beams else [self.bos_id, self.eos_id])

        # Strip BOS / EOS
        if best_tokens and best_tokens[0]  == self.bos_id: best_tokens = best_tokens[1:]
        if best_tokens and best_tokens[-1] == self.eos_id: best_tokens = best_tokens[:-1]

        return self._post_process(self.tokenizer.decode(best_tokens), clean_src, tgt_lang)

    def _post_process(self, text: str, src: str, tgt_lang: str) -> str:
        """model space -> native script (toolkit) -> repetition clean-up -> sentence punctuation."""
        text = strip_unk(text.replace("▁", " "))
        text = normalize_text(strip_tags(text))
        text = self.indic.from_model([text], tgt_lang)[0]
        return postprocess_output(text, src=src, lang=tgt_lang,
                                  digits=self.config.digits, latin=self.config.latin_policy)


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """Same normalisation as training data (NFC, spacing, danda), tags removed."""
    text = strip_unk(text.replace("▁", " "))
    return normalize_text(strip_tags(text))      # digits -> ASCII, so metrics ignore digit style


def show_text(text: str, lang: str, config: EvalConfig) -> str:
    """Text as a person should read it in the report files (native digits if enabled)."""
    text = normalize_text(strip_unk(text), digits='keep')
    return localize_digits(text, lang) if config.digits == 'native' else text


# ============================================================================
# INDIVIDUAL METRIC FUNCTIONS
# ============================================================================

def compute_cer(predictions, references):
    """Character Error Rate — lower is better."""
    if not HAS_EDITDISTANCE:
        return None
    total_errors = sum(editdistance.eval(p, r) for p, r in zip(predictions, references))
    total_chars  = sum(len(r) for r in references)
    return round(total_errors / total_chars * 100, 2) if total_chars > 0 else 0.0


def compute_wer(predictions, references):
    """Word Error Rate — lower is better."""
    if not HAS_EDITDISTANCE:
        return None
    total_errors = sum(editdistance.eval(p.split(), r.split())
                       for p, r in zip(predictions, references))
    total_words  = sum(len(r.split()) for r in references)
    return round(total_errors / total_words * 100, 2) if total_words > 0 else 0.0


def compute_char_f1(predictions, references):
    """Character-level F1 — robust to OOV and morphological variation."""
    scores = []
    for pred, ref in zip(predictions, references):
        pc, rc = set(pred.lower()), set(ref.lower())
        if not pc and not rc:
            scores.append(1.0); continue
        if not pc or not rc:
            scores.append(0.0); continue
        tp        = len(pc & rc)
        precision = tp / len(pc)
        recall    = tp / len(rc)
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return round(sum(scores) / len(scores) * 100, 2) if scores else 0.0


def compute_lebleu(predictions, references, max_n=4):
    """LeBLEU: character n-gram BLEU, better for morphologically rich languages."""
    import math
    from collections import Counter

    def ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

    scores = []
    for pred, ref in zip(predictions, references):
        pc, rc = list(pred), list(ref)
        precisions = []
        for n in range(1, min(max_n + 1, len(pc) + 1)):
            pred_ng = Counter(ngrams(pc, n))
            ref_ng  = Counter(ngrams(rc, n))
            matches = sum((pred_ng & ref_ng).values())
            total   = sum(pred_ng.values())
            if total > 0:
                precisions.append(matches / total)
        if precisions:
            geo_mean = math.exp(
                sum(math.log(p) for p in precisions if p > 0) / len(precisions)
            )
            bp = 1.0 if len(pc) >= len(rc) else math.exp(1 - len(rc) / max(len(pc), 1))
            scores.append(bp * geo_mean)
    return round(100 * sum(scores) / len(scores), 2) if scores else 0.0


def compute_length_stats(predictions, references):
    """Average length and correlation between prediction and reference lengths."""
    from scipy.stats import pearsonr, spearmanr

    pred_lens = [len(p.split()) for p in predictions]
    ref_lens  = [len(r.split()) for r in references]

    avg_pred = np.mean(pred_lens)
    avg_ref  = np.mean(ref_lens)
    ratio    = avg_pred / avg_ref if avg_ref > 0 else 0.0

    try:
        pearson_r,  _ = pearsonr(pred_lens,  ref_lens)
        spearman_r, _ = spearmanr(pred_lens, ref_lens)
    except Exception:
        pearson_r = spearman_r = 0.0

    return {
        'avg_pred_length': round(avg_pred, 2),
        'avg_ref_length':  round(avg_ref,  2),
        'length_ratio':    round(ratio,     3),
        'pearson_r':       round(pearson_r,  3),
        'spearman_r':      round(spearman_r, 3),
    }


def compute_sentence_bleu(predictions, references):
    """Sentence-level BLEU list (used for error analysis)."""
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            scores.append(sacrebleu.sentence_bleu(pred, [ref], tokenize="13a").score)
        except Exception:
            scores.append(0.0)
    return scores


def compute_bootstrap_ci(predictions, references, n_samples=1000):
    """Bootstrap 95% confidence interval for corpus BLEU."""
    n = len(predictions)
    bleu_scores = []
    for _ in tqdm(range(n_samples), desc="  Bootstrap CI", leave=False):
        idx = np.random.choice(n, n, replace=True)
        preds = [predictions[i] for i in idx]
        refs  = [[references[i]] for i in idx]
        try:
            bleu_scores.append(
                sacrebleu.corpus_bleu(preds, refs, tokenize="13a").score
            )
        except Exception:
            continue
    arr = np.array(bleu_scores)
    return {
        'mean':     round(float(np.mean(arr)),              2),
        'ci_lower': round(float(np.percentile(arr,  2.5)), 2),
        'ci_upper': round(float(np.percentile(arr, 97.5)), 2),
        'stderr':   round(float(np.std(arr)),               2),
    }


# ============================================================================
# MASTER METRICS FUNCTION
# ============================================================================

def compute_metrics(predictions, references, sources=None, config: EvalConfig = None):
    """
    Compute all enabled metrics and return a results dict.

    Args:
        predictions : list[str]
        references  : list[str]
        sources     : list[str] | None  (needed for COMET only)
        config      : EvalConfig

    Returns:
        dict  {metric_name: value}
    """
    print("\n📊 Computing metrics...")
    results = {}

    pred_clean = [clean_text(p) for p in predictions]
    ref_clean  = [[clean_text(r)] for r in references]
    ref_single = [clean_text(r)   for r in references]

    print(f"  Predictions: {len(pred_clean)}")
    empty = sum(1 for p in pred_clean if not p.strip())
    if empty:
        print(f"  ⚠️  {empty} empty predictions!")

    # ------------------------------------------------------------------
    # STANDARD METRICS
    # ------------------------------------------------------------------
    print("\n📐 Standard Metrics:")

    # BLEU 13a
    try:
        b = sacrebleu.corpus_bleu(pred_clean, ref_clean, tokenize="13a")
        results["BLEU_13a"] = round(b.score, 2)
        print(f"    ✓ BLEU (13a): {results['BLEU_13a']}")
    except Exception as e:
        print(f"    ✗ BLEU (13a) failed: {e}")
        results["BLEU_13a"] = None

    # BLEU no-tok (diagnostic)
    try:
        b2 = sacrebleu.corpus_bleu(pred_clean, ref_clean, tokenize="none")
        results["BLEU_none"] = round(b2.score, 2)
        print(f"    ✓ BLEU (none): {results['BLEU_none']}")
        if results["BLEU_13a"] and results["BLEU_none"]:
            diff = abs(results["BLEU_13a"] - results["BLEU_none"])
            if diff > 3.0:
                print(f"    ⚠️  Large BLEU diff ({diff:.2f}) — check preprocessing!")
    except Exception as e:
        print(f"    ✗ BLEU (none) failed: {e}")
        results["BLEU_none"] = None

    # spBLEU (FLORES-200 SentencePiece tokenisation - the usual choice for Indic languages;
    # needs a one-off download of the tokenizer model, skipped silently if offline)
    try:
        sb = sacrebleu.corpus_bleu(pred_clean, ref_clean, tokenize="flores200")
        results["spBLEU"] = round(sb.score, 2)
        print(f"    ✓ spBLEU: {results['spBLEU']}")
    except Exception as e:
        print(f"    - spBLEU skipped ({type(e).__name__})")
        results["spBLEU"] = None

    # chrF
    try:
        results["chrF"] = round(sacrebleu.corpus_chrf(pred_clean, ref_clean).score, 2)
        print(f"    ✓ chrF: {results['chrF']}")
    except Exception as e:
        print(f"    ✗ chrF failed: {e}")
        results["chrF"] = None

    # chrF++
    try:
        chrf_pp = sacrebleu.CHRF(word_order=2)
        results["chrF++"] = round(chrf_pp.corpus_score(pred_clean, ref_clean).score, 2)
        print(f"    ✓ chrF++: {results['chrF++']} ")
    except Exception as e:
        print(f"    ✗ chrF++ failed: {e}")
        results["chrF++"] = None

    # TER
    try:
        results["TER"] = round(sacrebleu.corpus_ter(pred_clean, ref_clean).score, 2)
        print(f"    ✓ TER: {results['TER']}")
    except Exception as e:
        print(f"    ✗ TER failed: {e}")
        results["TER"] = None

    # METEOR
    if HAS_EVALUATE and config and config.compute_meteor:
        try:
            meteor_metric = evaluate.load("meteor")
            score = meteor_metric.compute(
                predictions=pred_clean, references=ref_single
            )['meteor']
            if score > 1.0:
                print(f"    ✗ METEOR > 1.0 ({score:.4f}) — impossible value, skipping")
                results["METEOR"] = None
            else:
                results["METEOR"] = round(score, 4)
                print(f"    ✓ METEOR: {results['METEOR']}")
        except Exception as e:
            print(f"    ✗ METEOR failed: {e}")
            results["METEOR"] = None

    # ------------------------------------------------------------------
    # LOW-RESOURCE SPECIFIC METRICS
    # ------------------------------------------------------------------
    print("\n🌍 Low-Resource Metrics:")

    if config and config.compute_cer:
        cer = compute_cer(pred_clean, ref_single)
        results["CER"] = cer
        print(f"    {'✓' if cer is not None else '✗'} CER: "
              f"{cer}%" if cer is not None else "    ✗ CER: install editdistance")

    if config and config.compute_wer:
        wer = compute_wer(pred_clean, ref_single)
        results["WER"] = wer
        print(f"    {'✓' if wer is not None else '✗'} WER: "
              f"{wer}%" if wer is not None else "    ✗ WER: install editdistance")

    if config and config.compute_char_f1:
        cf1 = compute_char_f1(pred_clean, ref_single)
        results["Char_F1"] = cf1
        print(f"    ✓ Char F1: {cf1}")

    if config and config.compute_lebleu:
        lb = compute_lebleu(pred_clean, ref_single)
        results["LeBLEU"] = lb
        print(f"    ✓ LeBLEU: {lb}")

    # ------------------------------------------------------------------
    # LENGTH ANALYSIS
    # ------------------------------------------------------------------
    print("\n📏 Length Analysis:")
    try:
        length_stats = compute_length_stats(pred_clean, ref_single)
        for k, v in length_stats.items():
            results[f"Length_{k}"] = v
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"    ✗ Length stats failed: {e}")

    # ------------------------------------------------------------------
    # CONFIDENCE INTERVALS
    # ------------------------------------------------------------------
    if config and config.compute_confidence_intervals and len(predictions) >= 100:
        print(f"\n📉 Bootstrap CI ({config.num_bootstrap_samples} samples):")
        ci = compute_bootstrap_ci(pred_clean, ref_single, config.num_bootstrap_samples)
        results["BLEU_CI_lower"] = ci['ci_lower']
        results["BLEU_CI_upper"] = ci['ci_upper']
        results["BLEU_stderr"]   = ci['stderr']
        print(f"    95% CI : [{ci['ci_lower']}, {ci['ci_upper']}]")
        print(f"    Std Err: {ci['stderr']}")

    # ------------------------------------------------------------------
    # OPTIONAL HEAVY METRICS
    # ------------------------------------------------------------------

    # BERTScore
    if HAS_EVALUATE and config and config.compute_bertscore:
        print("\n  Computing BERTScore (slow)...")
        try:
            bs = evaluate.load("bertscore")
            result = bs.compute(
                predictions=pred_clean,
                references=ref_single,
                model_type="xlm-roberta-base",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            avg_f1 = sum(result['f1']) / len(result['f1'])
            results["BERTScore_F1"] = round(avg_f1, 4)
            print(f"    ✓ BERTScore F1: {results['BERTScore_F1']}")
        except Exception as e:
            print(f"    ✗ BERTScore failed: {e}")
            results["BERTScore_F1"] = None

    print("\n  ✅ Metrics done!\n")
    return results


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def save_error_analysis(predictions, references, sources, output_dir: Path,
                        tag: str, config: EvalConfig, src_lang: str, tgt_lang: str):
    """Write worst-translation file and (optionally) sentence-level scores."""
    if not config.save_worst_translations:
        return

    sent_bleus   = compute_sentence_bleu([clean_text(p) for p in predictions],
                                         [clean_text(r) for r in references])
    worst_idx    = np.argsort(sent_bleus)[:config.num_worst_examples]

    err_file = output_dir / f"error_analysis_{tag}.txt"
    with open(err_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ERROR ANALYSIS: {tag}\n")
        f.write(f"Worst {config.num_worst_examples} translations (by sentence BLEU)\n")
        f.write("=" * 80 + "\n\n")
        for rank, idx in enumerate(worst_idx, 1):
            f.write(f"Rank {rank} | Sent-BLEU: {sent_bleus[idx]:.2f}\n")
            f.write(f"  Source    : {show_text(sources[idx], src_lang, config)}\n")
            f.write(f"  Reference : {show_text(references[idx], tgt_lang, config)}\n")
            f.write(f"  Prediction: {show_text(predictions[idx], tgt_lang, config)}\n")
            f.write("-" * 80 + "\n\n")

    print(f"  ✅ Error analysis: {err_file.name}")

    if config.save_sentence_level_scores:
        sc_file = output_dir / f"sentence_scores_{tag}.txt"
        with open(sc_file, "w", encoding="utf-8") as f:
            f.write("sentence_id\tsentence_bleu\n")
            for i, s in enumerate(sent_bleus):
                f.write(f"{i}\t{s:.2f}\n")
        print(f"  ✅ Sentence scores: {sc_file.name}")


# ============================================================================
# EVALUATE ONE DIRECTION
# ============================================================================

def evaluate_direction(decoder: BeamSearchDecoder, src_texts, ref_texts,
                       tag: str, output_dir: Path, config: EvalConfig,
                       src_lang: str, tgt_lang: str):
    """Decode all source sentences and compute metrics for one direction."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {tag}")
    print(f"{'='*60}")

    predictions = []
    for src in tqdm(src_texts, desc=f"Decoding {tag}"):
        try:
            predictions.append(decoder.decode(src, src_lang, tgt_lang))
        except Exception:
            predictions.append("")

    # References are compared in the same canonical form the predictions are produced in
    # (identity when the toolkit is off) so formatting-only differences cost nothing.
    ref_texts = decoder.indic.canonical(ref_texts, tgt_lang)

    # Save raw predictions
    pred_file = output_dir / f"predictions_{tag}.txt"
    pred_file.write_text("\n".join(p.strip() for p in predictions), encoding="utf-8")

    # Save side-by-side comparison (first 100)
    comp_file = output_dir / f"comparison_{tag}.txt"
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write(f"{'Source':<50} | {'Reference':<50} | {'Prediction':<50}\n")
        f.write("-" * 155 + "\n")
        for src, ref, pred in zip(src_texts[:100], ref_texts[:100], predictions[:100]):
            f.write(f"{show_text(src, src_lang, config)[:50]:<50} | "
                    f"{show_text(ref, tgt_lang, config)[:50]:<50} | "
                    f"{show_text(pred, tgt_lang, config)[:50]:<50}\n")

    # Metrics
    results = compute_metrics(
        predictions=predictions,
        references=ref_texts,
        sources=src_texts,
        config=config,
    )

    # Error analysis
    save_error_analysis(predictions, ref_texts, src_texts, output_dir, tag, config, src_lang, tgt_lang)

    # Console summary
    print("\n📈 Results:")
    for metric in ['BLEU_13a', 'chrF++', 'CER', 'WER', 'Char_F1', 'LeBLEU', 'METEOR']:
        if metric in results and results[metric] is not None:
            print(f"  {metric}: {results[metric]}")

    return results, predictions


# ============================================================================
# DATA LOADING
# ============================================================================

def load_text_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [normalize_text(l) for l in f if l.strip()]


def load_test_pair(l1_path: str, l2_path: str):
    a, b = load_text_file(l1_path), load_text_file(l2_path)
    n = min(len(a), len(b))
    return a[:n], b[:n]


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=EvalConfig.run_name)
    ap.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=None,
                    help="default: pairs the run was trained on (from run name)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--beam-size", type=int, default=EvalConfig.beam_size)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--digits", choices=["native", "ascii"], default=EvalConfig.digits,
                    help="write numbers in the target script's own digits (default) or ASCII")
    ap.add_argument("--latin", choices=list(LATIN_POLICIES), default=EvalConfig.latin_policy,
                    help="invented English words: source-only (default) | keep | drop")
    args = ap.parse_args()

    config = EvalConfig()
    config.run_name        = args.run_name
    config.pairs           = args.pairs or [p for p in args.run_name.split("+") if p in PAIRS]
    config.checkpoint_path = args.checkpoint or f"checkpoints/{args.run_name}/best_model.pt"
    config.tokenizer_path  = f"tokenizer/{args.run_name}/spm.model"
    config.test_dir        = f"processed/{args.run_name}"
    config.beam_size       = args.beam_size
    config.digits          = args.digits
    config.latin_policy    = args.latin
    config.output_dir      = args.output_dir or f"evaluation_results/{args.run_name}"

    print("\n" + "=" * 70)
    print(f"  TRANSFORMER MT - EVALUATION  ({args.run_name})")
    print("=" * 70)

    preproc_file = Path(config.test_dir) / "preproc.json"
    indic_mode, meta = "off", {}
    if preproc_file.exists():
        meta = json.loads(preproc_file.read_text(encoding="utf-8"))
        indic_mode = meta.get("indic_mode", "off")
    else:
        print("⚠️  preproc.json not found - assuming --indic-mode off")
    if not args.pairs and meta.get("pairs"):            # pairs the run was trained on (works with any --run-name)
        config.pairs = [p for p in meta["pairs"] if p in PAIRS]
    if not config.pairs:
        raise SystemExit("No language pairs to evaluate: pass --pairs asm-brx and/or ben-odi")
    indic = IndicText(indic_mode)
    print(f"🔤 Indic mode (from training): {indic_mode}")

    tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    print(f"\n✅ Tokenizer: vocab={tokenizer.vocab_size()}")

    model   = load_model(config.checkpoint_path, config.device)
    decoder = BeamSearchDecoder(model, tokenizer, config, indic)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results, labels = {}, {}
    for pair_key in config.pairs:
        l1, l2 = PAIRS[pair_key]["langs"]
        t1, t2 = load_test_pair(f"{config.test_dir}/{pair_key}.test.{l1}.txt",
                                f"{config.test_dir}/{pair_key}.test.{l2}.txt")
        print(f"\n📊 {pair_key}: {len(t1)} test sentence pairs")

        for src_lang, tgt_lang, src_txt, ref_txt in ((l1, l2, t1, t2), (l2, l1, t2, t1)):
            tag = f"{src_lang}2{tgt_lang}"
            res, _ = evaluate_direction(
                decoder, src_txt, ref_txt, tag, output_dir, config, src_lang, tgt_lang)
            all_results[tag] = res
            labels[tag] = f"{LANGS[src_lang]['name']}→{LANGS[tgt_lang]['name']}"

    # ── Summary report ─────────────────────────────────────────────────
    metrics_order = ['BLEU_13a', 'spBLEU', 'chrF++', 'TER', 'CER', 'WER', 'Char_F1', 'LeBLEU']
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n  TRANSFORMER MT - EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        f.write(f"Checkpoint: {config.checkpoint_path}\n")
        for tag, results in all_results.items():
            f.write(f"\n{labels[tag]}:\n" + "-" * 80 + "\n")
            for m in metrics_order + ['BLEU_none', 'chrF', 'METEOR']:
                if results.get(m) is not None:
                    f.write(f"  {m:.<25} {results[m]}\n")
            for k, v in results.items():
                if k.startswith("Length_"):
                    f.write(f"  {k.replace('Length_', ''):.<25} {v}\n")
            if 'BLEU_CI_lower' in results:
                f.write(f"  95% CI (BLEU)............ [{results['BLEU_CI_lower']}, "
                        f"{results['BLEU_CI_upper']}]  stderr {results['BLEU_stderr']}\n")

        f.write("\n" + "=" * 80 + "\nSUMMARY TABLE\n" + "=" * 80 + "\n")
        f.write(f"{'Metric':<10}" + "".join(f" {labels[t]:<22}" for t in all_results) + "\n")
        for m in metrics_order:
            f.write(f"{m:<10}" + "".join(
                f" {str(all_results[t].get(m, 'N/A')):<22}" for t in all_results) + "\n")
        f.write("\nBLEU/chrF++ guide for low-resource MT: BLEU>20 good, 10-20 moderate, <10 weak; "
                "chrF++ is usually 10-15 points higher and fairer for agglutinative languages.\n")

    print(f"\n{'='*70}\nEVALUATION COMPLETE\n{'='*70}")
    print(f"{'Direction':<26} {'BLEU':>6} {'spBLEU':>7} {'chrF++':>7} {'CER':>6}")
    for t, r in all_results.items():
        print(f"{labels[t]:<26} {str(r.get('BLEU_13a','N/A')):>6} {str(r.get('spBLEU','N/A')):>7} "
              f"{str(r.get('chrF++','N/A')):>7} {str(r.get('CER','N/A')):>6}")
    print(f"\n📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
