"""
compute_metrics.py - Evaluation 

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
import json
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
from text_utils import LANGS, PAIRS, normalize_text
from indic_utils import IndicTransPreprocessor, INSTALL_HINT
from inference import DecodeConfig, load_model, BeamSearchDecoder, Translator
from postprocess import canonical


# ============================================================================
# CONFIGURATION  (decoding / post-processing options live in inference.DecodeConfig)
# ============================================================================

class EvalConfig(DecodeConfig):
    # Set from the command line in main():  --run-name / --pairs / --checkpoint
    run_name        = "asm-brx+ben-odi"
    pairs           = ["asm-brx", "ben-odi"]
    checkpoint_path = "/home/vikas1/vikas/vikas/icon_shared_ind2ind/scratch_moe/indic_mt2/checkpoints/indic_moe/checkpoint_epoch_13_early_stop.pt"     # default: checkpoints/<run_name>/best_model.pt
    tokenizer_path  = None     # default: spm model recorded in the checkpoint
    test_dir        = None     # default: processed/<data_name>   (<pair>.test.<lang>.txt files)

    use_moe = False
    output_dir = "evaluation_results"

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


# ============================================================================
# TEXT CLEANING (scoring form)
# ============================================================================

def clean_text(text: str) -> str:
    """Canonical SCORING form: artefacts / tags removed, NFC, native digits -> ASCII, straight
    quotes.  Every system (and the reference) goes through this, whatever digit style the
    saved predictions use."""
    return canonical(text)


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
                        tag: str, config: EvalConfig):
    """Write worst-translation file and (optionally) sentence-level scores."""
    if not config.save_worst_translations:
        return

    sent_bleus   = compute_sentence_bleu(predictions, references)
    worst_idx    = np.argsort(sent_bleus)[:config.num_worst_examples]

    err_file = output_dir / f"error_analysis_{tag}.txt"
    with open(err_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ERROR ANALYSIS: {tag}\n")
        f.write(f"Worst {config.num_worst_examples} translations (by sentence BLEU)\n")
        f.write("=" * 80 + "\n\n")
        for rank, idx in enumerate(worst_idx, 1):
            f.write(f"Rank {rank} | Sent-BLEU: {sent_bleus[idx]:.2f}\n")
            f.write(f"  Source    : {normalize_text(sources[idx])}\n")
            f.write(f"  Reference : {normalize_text(references[idx])}\n")
            f.write(f"  Prediction: {predictions[idx]}\n")
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

def evaluate_direction(translator: Translator, src_texts, ref_texts,
                       tag: str, output_dir: Path, config: EvalConfig,
                       src_lang: str, tgt_lang: str):
    """Decode all RAW source sentences (full pre -> model -> post pipeline) and score them."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {tag}")
    print(f"{'='*60}")

    raw_preds, predictions = [], []
    for src in tqdm(src_texts, desc=f"Decoding {tag}"):
        try:
            raw = translator.translate_raw(src, src_lang, tgt_lang)
            raw_preds.append(raw)
            predictions.append(translator.finalize(raw, src, tgt_lang))
        except Exception as e:
            print(f"   ⚠️ decoding failed ({type(e).__name__}: {e})")
            raw_preds.append("")
            predictions.append("")

    # Final predictions (native digits etc.) + the untouched decoder output for debugging
    (output_dir / f"predictions_{tag}.txt").write_text(
        "\n".join(p.strip() for p in predictions) + "\n", encoding="utf-8")
    (output_dir / f"predictions_{tag}.raw.txt").write_text(
        "\n".join(p.strip() for p in raw_preds) + "\n", encoding="utf-8")

    # Side-by-side comparison (first 100)
    with open(output_dir / f"comparison_{tag}.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Source':<50} | {'Reference':<50} | {'Prediction':<50}\n")
        f.write("-" * 155 + "\n")
        for src, ref, pred in zip(src_texts[:100], ref_texts[:100], predictions[:100]):
            f.write(f"{normalize_text(src)[:50]:<50} | {normalize_text(ref)[:50]:<50} | {pred[:50]:<50}\n")

    # Metrics (canonical form: digit style / nukta style do not matter)
    results = compute_metrics(predictions=predictions, references=ref_texts,
                              sources=src_texts, config=config)

    n_artifacts = sum('\u2047' in p or '<unk>' in p for p in raw_preds)
    results["raw_unk_outputs"] = n_artifacts
    if n_artifacts:
        print(f"  ℹ️  {n_artifacts} raw outputs contained <unk> (cleaned in the final predictions)")

    save_error_analysis(predictions, ref_texts, src_texts, output_dir, tag, config)

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
                    help="default: pairs stored in the checkpoint")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokenizer", default=None, help="default: spm model recorded in the checkpoint")
    ap.add_argument("--beam-size", type=int, default=EvalConfig.beam_size)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--digits", choices=["native", "ascii", "keep"], default="native",
                    help="digit style of the saved predictions (scores are unaffected)")
    ap.add_argument("--latin-policy", choices=["source", "keep", "drop"], default="source",
                    help="English words: keep only those found in the source (default) / keep all / drop all")
    ap.add_argument("--no-nukta-recompose", action="store_true")
    ap.add_argument("--length-penalty", type=float, default=EvalConfig.length_penalty)
    ap.add_argument("--max-sentences", type=int, default=None, help="evaluate only the first N test sentences")
    args = ap.parse_args()

    config = EvalConfig()
    config.run_name        = args.run_name
    config.checkpoint_path = args.checkpoint or f"checkpoints/{args.run_name}/best_model.pt"
    config.beam_size       = args.beam_size
    config.length_penalty  = args.length_penalty
    config.digits          = args.digits
    config.latin_policy    = args.latin_policy
    config.recompose_nukta = not args.no_nukta_recompose
    config.output_dir      = args.output_dir or f"evaluation_results/{args.run_name}"

    model, train_cfg = load_model(config.checkpoint_path, config.device)

    # everything about the data / pipeline comes from the checkpoint (fallbacks = old runs)
    data_name = train_cfg.get("data_name", args.run_name)
    config.pairs = (args.pairs or train_cfg.get("pairs")
                    or [p for p in args.run_name.split("+") if p in PAIRS])
    config.tokenizer_path    = (args.tokenizer or train_cfg.get("spm_model")
                                or f"tokenizer/{data_name}/spm.model")
    config.test_dir          = f"processed/{data_name}"
    config.use_indictrans    = bool(train_cfg.get("use_indictrans", False))
    config.indic_placeholders = bool(train_cfg.get("indic_placeholders", False))
    config.use_moe           = bool(model.moe_enabled)

    print("\n" + "=" * 70)
    print(f"  TRANSFORMER MT - EVALUATION  ({args.run_name})")
    print(f"  IndicTrans pre/post-processing : {'ON' if config.use_indictrans else 'off'}")
    print(f"  Mixture-of-Experts             : {'ON' if config.use_moe else 'off'}")
    print("=" * 70)

    tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    print(f"\n✅ Tokenizer: vocab={tokenizer.vocab_size()}  ({config.tokenizer_path})")

    indic = None
    if config.use_indictrans:
        try:
            indic = IndicTransPreprocessor(placeholders=config.indic_placeholders)
        except ImportError as e:
            raise SystemExit(f"❌ This checkpoint was trained with IndicTrans pre-processing.\n{e}")

    translator = Translator(model, tokenizer, config, indic)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results, labels = {}, {}
    for pair_key in config.pairs:
        l1, l2 = PAIRS[pair_key]["langs"]
        t1, t2 = load_test_pair(f"{config.test_dir}/{pair_key}.test.{l1}.txt",
                                f"{config.test_dir}/{pair_key}.test.{l2}.txt")
        if args.max_sentences:
            t1, t2 = t1[:args.max_sentences], t2[:args.max_sentences]
        print(f"\n📊 {pair_key}: {len(t1)} test sentence pairs")

        for src_lang, tgt_lang, src_txt, ref_txt in ((l1, l2, t1, t2), (l2, l1, t2, t1)):
            tag = f"{src_lang}2{tgt_lang}"
            res, _ = evaluate_direction(
                translator, src_txt, ref_txt, tag, output_dir, config,
                src_lang=src_lang, tgt_lang=tgt_lang)
            all_results[tag] = res
            labels[tag] = f"{LANGS[src_lang]['name']}→{LANGS[tgt_lang]['name']}"

    # ── Summary report ─────────────────────────────────────────────────
    metrics_order = ['BLEU_13a', 'spBLEU', 'chrF++', 'TER', 'CER', 'WER', 'Char_F1', 'LeBLEU']
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n  TRANSFORMER MT - EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        f.write(f"Checkpoint: {config.checkpoint_path}\n")
        f.write(f"IndicTrans pre/post-processing: {'ON' if config.use_indictrans else 'off'}\n")
        f.write(f"Mixture-of-Experts            : {'ON' if config.use_moe else 'off'}\n")
        f.write(f"Output post-processing        : digits={config.digits}, latin={config.latin_policy}, "
                f"nukta={'on' if config.recompose_nukta else 'off'}\n")
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

    (output_dir / "results.json").write_text(json.dumps(dict(
        run_name=config.run_name, checkpoint=config.checkpoint_path,
        indictrans=config.use_indictrans, moe=config.use_moe,
        digits=config.digits, latin_policy=config.latin_policy,
        parameters=model.get_num_params(), beam_size=config.beam_size,
        directions={t: dict(label=labels[t], metrics=r) for t, r in all_results.items()},
    ), indent=2, ensure_ascii=False, default=float), encoding="utf-8")

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
