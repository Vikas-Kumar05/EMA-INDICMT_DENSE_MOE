"""
text_utils.py - Language registry, text normalisation, script validation
and file loading shared by tokeniser.py, prepare_dataset.py, train.py
and compute_metrics.py.

Languages
=========
asm  Assamese  (Bengali-Assamese script, U+0980-09FF)
brx  Bodo      (Devanagari script,        U+0900-097F)
ben  Bengali   (Bengali-Assamese script, U+0980-09FF)
odi  Odia      (Odia script,              U+0B00-0B7F)

NOTE: Assamese and Bengali share one Unicode block, so the old
"reject anything in U+0980-09FF" filter would have deleted every row.
Validation is now per language: a sentence must be mostly written in the
script that language is expected to use.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LANGS = {
    'asm': dict(name='Assamese', tag='<2asm>', ranges=[(0x0980, 0x09FF)]),
    'brx': dict(name='Bodo',     tag='<2brx>', ranges=[(0x0900, 0x097F)]),
    'ben': dict(name='Bengali',  tag='<2ben>', ranges=[(0x0980, 0x09FF)]),
    'odi': dict(name='Odia',     tag='<2odi>', ranges=[(0x0B00, 0x0B7F)]),
}

# language-direction tokens (always the same 4, so token ids never change)
LANG_TAGS = [v['tag'] for v in LANGS.values()]

PAIRS = {
    'asm-brx': dict(langs=('asm', 'brx'), cols=('asm', 'bodo'),
                    stem='asm_bodo'),
    'ben-odi': dict(langs=('ben', 'odi'), cols=('bengali', 'odia'),
                    stem='bengali_odia'),
}

TAG_RE = re.compile('|'.join(re.escape(t) for t in LANG_TAGS))


def strip_tags(text):
    return TAG_RE.sub('', text)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# zero-width space, BOM, soft hyphen, LRM/RLM.  ZWJ (200D) and ZWNJ (200C)
# are deliberately KEPT: they change conjunct shaping in Indic scripts.
_INVISIBLE = re.compile('[​﻿­‎‏⁠]')
_SPACE_BEFORE_PUNCT = re.compile(r'\s+([।॥.,;?!])')
_TRAILING_PIPE = re.compile(r'\s*\|\s*$')
_HTML_ENTITY = re.compile(r'&(nbsp|amp|quot|lt|gt|#\d+);')
_REPEAT_PUNCT = re.compile(r'([।॥?!,;:])\1+')          # "।।" "??" "!!" -> single
_REPEAT_DOTS = re.compile(r'\.{4,}')                     # "......" -> "..."

# typographic variants -> one canonical ASCII form (same idea as the toolkit's
# punctuation normaliser, so toolkit on/off give the same surface forms)
_PUNCT_MAP = str.maketrans({
    '“': '"', '”': '"', '„': '"', '«': '"', '»': '"',
    '‘': "'", '’': "'", '‚': "'", 'ʼ': "'", '`': "'",
    '–': '-', '—': '-', '−': '-',
    '…': '...',
    ' ': ' ', ' ': ' ', ' ': ' ', '　': ' ',
    '，': ',', '？': '?', '！': '!', '：': ':', '；': ';',
})

# Bengali-Assamese, Devanagari, Odia digits -> ASCII.  The three scripts of this
# project write numbers in a mix of native and Latin digits; one form makes
# the native and Latin spelling of a number the same token for the model and BLEU.
_DIGITS = {}
for _base in (0x09E6, 0x0966, 0x0B66):
    for _i in range(10):
        _DIGITS[_base + _i] = str(_i)
_DIGIT_MAP = str.maketrans(_DIGITS)

# Bump this whenever normalisation / tokenizer settings change. main.py compares it with the
# value stored in processed/<run>/preproc.json and rebuilds data + tokenizer when they differ.
PREPROC_VERSION = 3

# ASCII digits -> native digits of each language (used on model OUTPUT only)
_NATIVE_BASE = {'asm': 0x09E6, 'ben': 0x09E6, 'brx': 0x0966, 'odi': 0x0B66}
_TO_NATIVE = {lang: str.maketrans({str(i): chr(base + i) for i in range(10)})
              for lang, base in _NATIVE_BASE.items()}
_ASCII_NUM = re.compile(r'[0-9]+(?:[.,][0-9]+)*')      # whole number: 14,033.12 / 0.06
_LATIN_CH = re.compile(r'[A-Za-z]')


def localize_digits(text, lang):
    """
    2024 -> ২০২৪ (asm/ben), २०२४ (brx), ୨୦୨୪ (odi).
    A number glued to a Latin letter (CO2, 3D, 400K, 0.06eV) is left alone: it is part of a
    formula or unit, not a native-script number.
    """
    table = _TO_NATIVE.get(lang)
    if table is None:
        return text

    def repl(m):
        a, b = m.start(), m.end()
        if (a > 0 and _LATIN_CH.match(text[a - 1])) or (b < len(text) and _LATIN_CH.match(text[b])):
            return m.group(0)
        return m.group(0).translate(table)
    return _ASCII_NUM.sub(repl, text)


# What SentencePiece prints for pieces outside its vocabulary ("⁇") and for broken UTF-8 bytes.
_UNK_JUNK = re.compile('[\u2047\ufffd]|<unk>')
_EMPTY_BRACKETS = re.compile(r'\(\s*[,;:]?\s*\)|\[\s*\]')
_BRACKET_SPACE = re.compile(r'([(\[])\s+|\s+([)\]])')
_REPEAT_SEP = re.compile(r'([,;:])(?:\s*\1)+')


def strip_unk(text):
    return _UNK_JUNK.sub('', text)

# Indic-NLP-Library normaliser (nukta composition, vowel-sign order, ...).
# NFC does NOT merge e.g. Bengali ড + ় into ড়, so this is a real gain.
_ISO = {'asm': 'as', 'brx': 'hi', 'ben': 'bn', 'odi': 'or'}
_INDIC_NORMALIZERS = {}
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    HAS_INDICNLP = True
except Exception:                                        # library not installed
    IndicNormalizerFactory = None
    HAS_INDICNLP = False


def _indic_normalizer(lang):
    if not HAS_INDICNLP:
        return None
    if lang not in _INDIC_NORMALIZERS:
        try:
            _INDIC_NORMALIZERS[lang] = IndicNormalizerFactory().get_normalizer(_ISO[lang])
        except Exception:
            _INDIC_NORMALIZERS[lang] = None
    return _INDIC_NORMALIZERS[lang]


def normalize_text(text, lang=None, digits='ascii'):
    """
    Apply the same cleaning to train / dev / test / inference text.
    Pass `lang` (asm/brx/ben/odi) to also run the Indic-NLP normaliser.
    digits='ascii' (default, used for all training data) converts native digits to ASCII;
    digits='keep' leaves them as they are (used when showing output to people).
    """
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFC', text)
    text = _INVISIBLE.sub('', text)
    text = _HTML_ENTITY.sub(' ', text)
    # control characters (incl. stray \r \n \t inside spreadsheet cells) -> space
    text = ''.join(' ' if unicodedata.category(c) == 'Cc' else c for c in text)
    text = text.translate(_PUNCT_MAP)
    if digits == 'ascii':
        text = text.translate(_DIGIT_MAP)
    text = _TRAILING_PIPE.sub('।', text)          # ASCII '|' used as danda
    text = _REPEAT_PUNCT.sub(r'\1', text)
    text = _REPEAT_DOTS.sub('...', text)
    if lang == 'odi':
        text = text.replace('\u0b2f\u0b3c', '\u0b5f')    # ଯ + nukta -> ୟ (one form for ya)
    if lang is not None:
        norm = _indic_normalizer(lang)
        if norm is not None:
            text = norm.normalize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r'\1', text)   # "word ।" -> "word।"
    return text


# ---------------------------------------------------------------------------
# Noise filters (used by tokeniser.clean_pair_df)
# ---------------------------------------------------------------------------

_URL_EMAIL_HTML = re.compile(r'(https?://|www\.|\S+@\S+\.\S+|</?[a-zA-Z][^>]*>)')


def noise_reason(text, max_words=150):
    """Return a short reason string if `text` looks like noise, else None."""
    if _URL_EMAIL_HTML.search(text):
        return 'url_html'
    words = text.split()
    if len(words) > max_words:
        return 'too_long'
    chars = [c for c in text if not c.isspace()]
    if chars and sum(c.isdigit() for c in chars) / len(chars) > 0.5:
        return 'mostly_digits'
    if len(words) >= 6:
        top = max(words.count(w) for w in set(words))
        if top / len(words) > 0.5:
            return 'repetitive'
    return None


# ---------------------------------------------------------------------------
# Output post-processing (used by compute_metrics.py after decoding)
# ---------------------------------------------------------------------------

_TERMINAL = '।॥.?!'


def collapse_repetitions(text, max_n=4, min_repeats=3):
    """
    Remove degenerate loops that beam search produces ("word word word word",
    or "a b a b a b").  A run of >= min_repeats identical n-grams is cut to one.
    Genuine doubled words (2 repeats, e.g. reduplication) are left untouched.
    """
    words = text.split()
    for n in range(1, max_n + 1):
        i, out = 0, []
        while i < len(words):
            gram = words[i:i + n]
            reps = 1
            while len(gram) == n and words[i + reps * n:i + (reps + 1) * n] == gram:
                reps += 1
            if len(gram) == n and reps >= min_repeats:
                out.extend(gram)
                i += reps * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return ' '.join(words)


def match_terminal_punct(pred, src):
    """If the source ends with sentence punctuation and the output lost it, restore it."""
    pred, src = pred.rstrip(), src.rstrip()
    if not pred or not src or pred[-1] in _TERMINAL + ',;:-':   # already ended, or cut off mid-clause
        return pred
    if src[-1] in _TERMINAL:
        return pred + src[-1] if src[-1] in '?!' else pred + '।'
    return pred


_LATIN_TOKEN = re.compile(r'[A-Za-z][A-Za-z0-9\-]*')
LATIN_POLICIES = ('source-only', 'keep', 'drop')


def _latin_set(src):
    toks = set()
    for t in _LATIN_TOKEN.findall(src or ''):
        t = t.lower()
        toks.add(t)
        toks.update(p for p in t.split('-') if p)
    return toks


def filter_latin(text, src, policy='source-only'):
    """
    The model sometimes invents English / Latin words ("fapulation", "Musiver Histor Buding").
      keep         : leave everything
      source-only  : keep a Latin word only if the SOURCE sentence contains it (acronyms, units,
                     formulas and English terms that are legitimately copied); drop the rest
      drop         : remove every Latin word
    """
    if policy == 'keep':
        return text
    if policy not in LATIN_POLICIES:
        raise ValueError(f"latin policy must be one of {LATIN_POLICIES}")
    allowed = _latin_set(src) if policy == 'source-only' else set()

    def repl(m):
        t = m.group(0).lower()
        return m.group(0) if (t in allowed or all(p in allowed for p in t.split('-') if p)) and allowed else ''
    return _LATIN_TOKEN.sub(repl, text)


def postprocess_output(text, src=None, lang=None, digits='native', latin='source-only'):
    """
    Clean a decoded sentence (already in the target language's native script):
      1. drop <unk> junk, normalise (digits are ASCII while working)
      2. remove invented Latin words (see filter_latin) and the empty brackets they leave
      3. collapse repetition loops, restore sentence-final punctuation
      4. digits='native': write numbers in the target script's own digits
    """
    text = strip_unk(text)
    text = normalize_text(text, lang)
    text = filter_latin(text, src, latin)
    text = _EMPTY_BRACKETS.sub('', text)
    text = _BRACKET_SPACE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _REPEAT_SEP.sub(r'\1', text)
    text = normalize_text(text, lang)
    text = collapse_repetitions(text)
    if src is not None:
        text = match_terminal_punct(text, src)
    if digits == 'native' and lang is not None:
        text = localize_digits(text, lang)
    return text


# ---------------------------------------------------------------------------
# Script validation
# ---------------------------------------------------------------------------

def script_ratio(text, lang):
    """Fraction of alphabetic characters that belong to lang's script."""
    ranges = LANGS[lang]['ranges']
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hit = sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters)
    return hit / len(letters)


def is_valid(text, lang, min_ratio=0.5):
    return bool(text) and script_ratio(text, lang) >= min_ratio


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def read_table(path):
    """Read .csv / .tsv / .xlsx / .xls into a DataFrame of strings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext in ('.xlsx', '.xlsm', '.xls'):
        df = pd.read_excel(path, dtype=str)
    elif ext == '.tsv':
        df = pd.read_csv(path, sep='\t', dtype=str, quoting=3)
    else:
        df = pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(df, wanted):
    """Case-insensitive column lookup."""
    lookup = {c.lower(): c for c in df.columns}
    if wanted.lower() not in lookup:
        raise ValueError(f"Column '{wanted}' not found. Available: {list(df.columns)}")
    return lookup[wanted.lower()]
