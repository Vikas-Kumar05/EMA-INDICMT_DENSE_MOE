"""
text_utils.py 

"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# 'flores'     = FLORES-200 code used by the IndicTrans toolkit (indic_utils.py)
# 'digit_zero' = code point of the native digit 0 (digits 0-9 are consecutive), used by
#                postprocess.py to write numbers in the target language's own script
LANGS = {
    'asm': dict(name='Assamese', tag='<2asm>', ranges=[(0x0980, 0x09FF)], flores='asm_Beng', digit_zero=0x09E6),
    'brx': dict(name='Bodo',     tag='<2brx>', ranges=[(0x0900, 0x097F)], flores='brx_Deva', digit_zero=0x0966),
    'ben': dict(name='Bengali',  tag='<2ben>', ranges=[(0x0980, 0x09FF)], flores='ben_Beng', digit_zero=0x09E6),
    'odi': dict(name='Odia',     tag='<2odi>', ranges=[(0x0B00, 0x0B7F)], flores='ory_Orya', digit_zero=0x0B66),
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
_INVISIBLE = re.compile('[\u200b\ufeff\u00ad\u200e\u200f\u2060]')
_SPACE_BEFORE_PUNCT = re.compile(r'\s+([।॥.,;?!])')
_TRAILING_PIPE = re.compile(r'\s*\|\s*$')


def normalize_text(text):
    """Apply the same cleaning to train / dev / test / inference text."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFC', text)
    text = _INVISIBLE.sub('', text)
    # control characters (incl. stray \r \n \t inside spreadsheet cells) -> space
    text = ''.join(' ' if unicodedata.category(c) == 'Cc' else c for c in text)
    text = _TRAILING_PIPE.sub('।', text)          # ASCII '|' used as danda
    text = re.sub(r'\s+', ' ', text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r'\1', text)   # "word ।" -> "word।"
    return text


# ---------------------------------------------------------------------------
# Canonical form used ONLY for scoring
# ---------------------------------------------------------------------------
# The IndicTrans pipeline writes ASCII digits and straight quotes, while raw corpora
# often contain native digits / curly quotes.  Scoring every system (baseline included)
# on this canonical form keeps the comparison fair.

_DIGIT_BLOCKS = [0x0966, 0x09E6, 0x0B66]          # Devanagari, Bengali-Assamese, Odia zero
_EVAL_TABLE = {base + i: ord('0') + i for base in _DIGIT_BLOCKS for i in range(10)}
_EVAL_TABLE.update({ord(c): "'" for c in "`´‘‚’"})
_EVAL_TABLE.update({ord(c): '"' for c in "„“”«»"})
_EVAL_TABLE.update({ord(c): '-' for c in "–—"})


def eval_normalize(text):
    """Native digits -> ASCII, curly quotes/dashes -> straight.  Scoring only."""
    return text.translate(_EVAL_TABLE)


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
