"""
tokeniser.py - 

"""

import os
import json
import string
from pathlib import Path

import pandas as pd
import sentencepiece as spm

from text_utils import (
    LANGS, LANG_TAGS, PAIRS, normalize_text, is_valid, script_ratio,
    read_table, find_column,
)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_pair_df(df, pair_key, min_script_ratio=0.5, max_char_ratio=3.5):
    """
    Return a DataFrame with columns [lang1, lang2] (language codes),
    cleaned and de-duplicated.
    """
    info = PAIRS[pair_key]
    l1, l2 = info['langs']
    c1 = find_column(df, info['cols'][0])
    c2 = find_column(df, info['cols'][1])

    out = pd.DataFrame({
        l1: df[c1].map(normalize_text),
        l2: df[c2].map(normalize_text),
    })
    n0 = len(out)
    stats = {}

    out = out[(out[l1].str.len() >= 2) & (out[l2].str.len() >= 2)]
    stats['empty'] = n0 - len(out)

    n = len(out)
    ok = (out[l1].map(lambda t: is_valid(t, l1, min_script_ratio)) &
          out[l2].map(lambda t: is_valid(t, l2, min_script_ratio)))
    out = out[ok]
    stats['wrong_script'] = n - len(out)

    n = len(out)
    ratio = out[l1].str.len() / out[l2].str.len()
    out = out[(ratio <= max_char_ratio) & (ratio >= 1 / max_char_ratio)]
    stats['length_ratio'] = n - len(out)

    n = len(out)
    out = out[out[l1] != out[l2]]
    stats['identical'] = n - len(out)

    n = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    stats['duplicates'] = n - len(out)

    print(f"   rows in file      : {n0:,}")
    for k, v in stats.items():
        print(f"   removed {k:<13}: {v:,}")
    print(f"   clean pairs       : {len(out):,}")
    return out


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _find_split_file(data_dir, stem, split):
    for ext in ('.csv', '.xlsx', '.tsv'):
        p = Path(data_dir) / f"{stem}_{split}{ext}"
        if p.exists():
            return p
    return None


def split_pair_df(df, dev_size, test_size, seed=42):
    """Random split; dev/test are capped at 5% each so train stays large."""
    n = len(df)
    dev_n  = min(dev_size,  max(1, int(0.05 * n)))
    test_n = min(test_size, max(1, int(0.05 * n)))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev   = df.iloc[:dev_n]
    test  = df.iloc[dev_n:dev_n + test_n]
    train = df.iloc[dev_n + test_n:]
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


def remove_leakage(train, held_out, l1, l2):
    """Drop training rows whose source OR target sentence is in dev/test."""
    seen1 = set(held_out[l1]); seen2 = set(held_out[l2])
    keep = ~(train[l1].isin(seen1) | train[l2].isin(seen2))
    removed = (~keep).sum()
    if removed:
        print(f"   removed {removed:,} train rows overlapping dev/test")
    return train[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bidirectional dataset
# ---------------------------------------------------------------------------

def make_bidirectional(df, l1, l2):
    """
    [l1, l2] -> [src, tgt] containing both directions.
    The source sentence is prefixed with the TARGET language tag, e.g.
        "<2brx> <assamese sentence>"  ->  <bodo sentence>
        "<2asm> <bodo sentence>"      ->  <assamese sentence>
    """
    fwd = pd.DataFrame({'src': LANGS[l2]['tag'] + ' ' + df[l1], 'tgt': df[l2]})
    bwd = pd.DataFrame({'src': LANGS[l1]['tag'] + ' ' + df[l2], 'tgt': df[l1]})
    return pd.concat([fwd, bwd], ignore_index=True)


# ---------------------------------------------------------------------------
# Full preparation
# ---------------------------------------------------------------------------

def to_model_space(df, l1, l2, indic):
    """Run the IndicTrans pre-processor over both columns; drop rows that become empty."""
    out = pd.DataFrame({
        l1: indic.preprocess_corpus(df[l1].tolist(), l1),
        l2: indic.preprocess_corpus(df[l2].tolist(), l2),
    })
    keep = (out[l1].str.len() >= 1) & (out[l2].str.len() >= 1)
    if (~keep).sum():
        print(f"   dropped {(~keep).sum():,} rows that became empty after IndicTrans preprocessing")
    return out[keep].reset_index(drop=True)


def prepare_all(pairs, data_dir='data', files=None, dev_files=None, test_files=None, out_dir='processed',
                dev_size=1000, test_size=1000, seed=42,
                min_script_ratio=0.5, max_char_ratio=3.5,
                indictrans=False, indic_placeholders=False):
    """
    Returns dict(train_csv, dev_csv, corpus, n_train) and writes all files to out_dir.

    files      : optional {pair_key: path} overriding <data_dir>/<stem>.csv|xlsx
    indictrans : apply the IndicTrans toolkit pre-processing to train/dev text
    """
    files = files or {}
    dev_files = dev_files or {}
    test_files = test_files or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indic = None
    if indictrans:
        from indic_utils import IndicTransPreprocessor
        indic = IndicTransPreprocessor(placeholders=indic_placeholders)
        print(f"\n🔧 IndicTrans toolkit pre-processing ON (placeholders={indic_placeholders})")

    train_parts, dev_parts, spm_text = [], [], []

    for pair_key in pairs:
        info = PAIRS[pair_key]
        l1, l2 = info['langs']
        print(f"\n{'='*60}\nPAIR {pair_key}  "
              f"({LANGS[l1]['name']} <-> {LANGS[l2]['name']})\n{'='*60}")

        path = files.get(pair_key)
        if path is None:
            for ext in ('.csv', '.xlsx', '.tsv'):
                cand = Path(data_dir) / f"{info['stem']}{ext}"
                if cand.exists():
                    path = cand
                    break
        if path is None:
            raise FileNotFoundError(
                f"No data file for {pair_key}. Put '{info['stem']}.csv' (or .xlsx) "
                f"in '{data_dir}' or pass an explicit path.")
        print(f"   file: {path}")

        df = clean_pair_df(read_table(path), pair_key, min_script_ratio, max_char_ratio)

        # sanity: warn if script purity looks low
        for lang in (l1, l2):
            r = df[lang].map(lambda t: script_ratio(t, lang)).mean()
            print(f"   mean {LANGS[lang]['name']} script purity: {r:.3f}")

        # dev / test: explicit path > <stem>_dev / <stem>_test file in data_dir > random split
        dev_f  = dev_files.get(pair_key)  or _find_split_file(data_dir, info['stem'], 'dev')
        test_f = test_files.get(pair_key) or _find_split_file(data_dir, info['stem'], 'test')

        train = df
        dev = test = None
        if dev_f:
            print(f"   dev  file: {dev_f}")
            dev = clean_pair_df(read_table(dev_f), pair_key, min_script_ratio, 1e9)
        if test_f:
            print(f"   test file: {test_f}")
            test = clean_pair_df(read_table(test_f), pair_key, min_script_ratio, 1e9)

        # anything not provided is split randomly off the training data
        if dev is None or test is None:
            train, dev_r, test_r = split_pair_df(df, dev_size, test_size, seed)
            if dev is None:
                dev = dev_r
            if test is None:
                test = test_r
            if dev is not dev_r and test is test_r:
                train = pd.concat([train, dev_r]).reset_index(drop=True)   # unused random dev goes back to train
            if test is not test_r and dev is dev_r:
                train = pd.concat([train, test_r]).reset_index(drop=True)  # unused random test goes back to train
        train = remove_leakage(train, pd.concat([dev, test]), l1, l2)
        print(f"   split -> train {len(train):,} | dev {len(dev):,} | test {len(test):,}")

        # plain-text test files for compute_metrics.py
        for lang in (l1, l2):
            (out_dir / f"{pair_key}.test.{lang}.txt").write_text(
                "\n".join(test[lang]) + "\n", encoding='utf-8')

        if indic is not None:
            print("   IndicTrans pre-processing train ...")
            train = to_model_space(train, l1, l2, indic)
            print("   IndicTrans pre-processing dev ...")
            dev = to_model_space(dev, l1, l2, indic)
            print(f"   e.g. {LANGS[l1]['name']}: {train[l1].iloc[0][:80]}")

        train_parts.append(make_bidirectional(train, l1, l2))
        dev_parts.append(make_bidirectional(dev, l1, l2))
        spm_text.extend(train[l1].tolist() + train[l2].tolist())

    train_df = pd.concat(train_parts, ignore_index=True)
    dev_df   = pd.concat(dev_parts,   ignore_index=True)
    train_csv = out_dir / 'bidir_train.csv'
    dev_csv   = out_dir / 'bidir_dev.csv'
    train_df.to_csv(train_csv, index=False, encoding='utf-8')
    dev_df.to_csv(dev_csv, index=False, encoding='utf-8')

    corpus = out_dir / 'joint_spm_corpus.txt'
    corpus.write_text("\n".join(spm_text) + "\n", encoding='utf-8')

    (out_dir / 'meta.json').write_text(json.dumps(dict(
        pairs=list(pairs), indictrans=bool(indictrans), indic_placeholders=bool(indic_placeholders),
        n_train=len(train_df), n_dev=len(dev_df), seed=seed)), encoding='utf-8')

    print(f"\n✅ bidirectional train: {len(train_df):,} | dev: {len(dev_df):,}")
    print(f"   SentencePiece corpus: {len(spm_text):,} lines (train text only)")
    return dict(train_csv=str(train_csv), dev_csv=str(dev_csv),
                corpus=str(corpus), n_train=len(train_df))


# ---------------------------------------------------------------------------
# SentencePiece
# ---------------------------------------------------------------------------

# Characters that must always be in the vocabulary, however rare they are in the corpus:
# Latin letters / digits / ASCII punctuation, Greek letters and common science symbols.
# Without them a single rare letter ("k", "W", "π", "°") becomes <unk>, which SentencePiece
# prints as " ⁇ " in the translation (e.g. "mar ⁇ er", "⁇ orld", "2 ⁇ r").
_GREEK = ''.join(chr(c) for c in list(range(0x391, 0x3AA)) + list(range(0x3B1, 0x3CA)))
_SYMBOLS = "°±×÷≈≠≤≥∞√∑∫∂∆∇→←↑↓↔⇒·•…–—‘’“”′″‰µΩ₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹€₹£¥©®™§¶"
REQUIRED_CHARS = string.ascii_letters + string.digits + string.punctuation + _GREEK + _SYMBOLS


def train_sentencepiece(input_file, model_prefix, vocab_size=16000,
                        character_coverage=0.9995, model_type='bpe', byte_fallback=True):
    """Joint tokenizer for all selected languages, with the 4 direction tags.

    byte_fallback=True : characters that are still unseen are split into UTF-8 byte pieces
                         instead of <unk>, so the model can never output "⁇" (adds 256 pieces).
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Training corpus not found: {input_file}")

    print(f"\n{'='*60}\nTraining SentencePiece ({model_type}, vocab={vocab_size:,})\n{'='*60}")
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    # BPE aborts ("freq > 0") on required characters that never occur in the corpus, so every
    # required character is also written once into a tiny extra input file.
    seed_file = Path(model_prefix).with_suffix('.required_chars.txt')
    seed_file.write_text(' '.join(REQUIRED_CHARS) + '\n', encoding='utf-8')

    spm.SentencePieceTrainer.Train(
        input=f"{input_file},{seed_file}",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,   # 0.9995 keeps rare Indic conjuncts, drops noise
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=LANG_TAGS,
        normalization_rule_name='identity',      # text is already NFC-normalised by text_utils; SPM's default NFKC would rewrite some Bengali/Odia letters
        split_digits=True,                       # digits appear in Latin, Bengali and Devanagari forms
        hard_vocab_limit=False,                  # shrink instead of crashing on small corpora
        byte_fallback=byte_fallback,             # rare symbols -> bytes, never <unk>
        required_chars=REQUIRED_CHARS,           # Latin / Greek / symbols always in the vocab
        num_threads=os.cpu_count() or 1,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
    )
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    print(f"✅ trained: {model_prefix}.model  (vocab {sp.vocab_size()}, byte_fallback={byte_fallback})")
    return sp


def verify_tokenizer(model_file):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Tokenizer model not found: {model_file}")
    sp = spm.SentencePieceProcessor(model_file=model_file)
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (0, 1, 2, 3), \
        "Special-token ids must be pad=0 unk=1 bos=2 eos=3"
    for t in LANG_TAGS:
        assert sp.piece_to_id(t) != sp.unk_id(), f"Language tag {t} missing from tokenizer"
    print(f"✅ Tokenizer verified: {model_file}  (vocab {sp.vocab_size()})")
    return sp


def report_unk_rate(sp, csv_file, n=5000):
    """Percent of tokens that are <unk> on a sample - tells you if coverage is OK."""
    df = pd.read_csv(csv_file, nrows=n)
    unk = tot = 0
    for text in list(df['src']) + list(df['tgt']):
        ids = sp.encode(str(text), out_type=int)
        unk += ids.count(sp.unk_id())
        tot += len(ids)
    print(f"   <unk> rate on sample: {100 * unk / max(tot, 1):.3f}%")
    if unk:
        bad = {}
        for text in list(df['src']) + list(df['tgt']):
            for ch in set(str(text)):
                if not ch.isspace() and sp.unk_id() in sp.encode(ch):
                    bad[ch] = bad.get(ch, 0) + 1
        top = sorted(bad.items(), key=lambda x: -x[1])[:15]
        print("   characters mapped to <unk>: " + "  ".join(f"{c!r}x{n}" for c, n in top)
              + "\n   -> train with byte_fallback (default) to remove them")


if __name__ == "__main__":
    info = prepare_all(list(PAIRS), data_dir='data', out_dir='processed')
    sp = train_sentencepiece(info['corpus'], 'tokenizer/spm_joint')
    report_unk_rate(sp, info['train_csv'])
