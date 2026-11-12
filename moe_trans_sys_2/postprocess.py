"""
postprocess.py 
"""

import re

from text_utils import LANGS, normalize_text, strip_tags, eval_normalize

# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------

_ARTEFACTS = re.compile('[\u2047\ufffd\ue000]|<unk>|<pad>|<s>|</s>')   # U+2047 = "⁇"


def strip_artifacts(text):
    """Remove <unk> renderings, SentencePiece word marks and language tags."""
    text = strip_tags(text.replace('\u2581', ' '))
    return _ARTEFACTS.sub('', text)


# ---------------------------------------------------------------------------
# digits
# ---------------------------------------------------------------------------

# a whole number: 14,033.12  /  0.06  /  2024   (not preceded by a letter or digit: H2O, B12)
_NUMBER = re.compile(r'(?<![A-Za-z0-9])[0-9]+(?:[.,][0-9]+)*')


def localize_digits(text, lang):
    """ASCII digits -> native digits of `lang`.  Numbers directly followed by a Latin letter
    (400K, 0.06eV, 5G) are left untouched so that unit expressions stay consistent."""
    zero = LANGS[lang]['digit_zero']
    table = {ord('0') + i: chr(zero + i) for i in range(10)}

    def repl(m):
        nxt = text[m.end():m.end() + 1]
        if nxt and nxt.isascii() and nxt.isalpha():
            return m.group(0)
        return m.group(0).translate(table)

    return _NUMBER.sub(repl, text)


def ascii_digits(text):
    """Native digits (any of the three scripts) -> ASCII."""
    return eval_normalize(text)


# ---------------------------------------------------------------------------
# nukta recomposition  (NFC decomposes these because they are composition exclusions)
# ---------------------------------------------------------------------------

_NUKTA = {
    'ben': [('\u09A1\u09BC', '\u09DC'), ('\u09A2\u09BC', '\u09DD'), ('\u09AF\u09BC', '\u09DF')],
    'odi': [('\u0B21\u0B3C', '\u0B5C'), ('\u0B22\u0B3C', '\u0B5D'), ('\u0B2F\u0B3C', '\u0B5F')],
}
_NUKTA['asm'] = _NUKTA['ben']          # Assamese shares the Bengali block


def recompose_nukta(text, lang):
    for old, new in _NUKTA.get(lang, ()):
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Latin-script (English) words
# ---------------------------------------------------------------------------

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'.][A-Za-z0-9]+)*")


def latin_words(text):
    return {w.lower() for w in _LATIN_WORD.findall(text)}


def filter_latin(text, source, policy='source'):
    """
    policy 'keep'   : leave Latin words alone
           'source' : keep a Latin word only if it also occurs in the source sentence
           'drop'   : remove every Latin word
    """
    if policy == 'keep':
        return text
    allowed = latin_words(source) if policy == 'source' else set()
    removed = []

    def repl(m):
        if m.group(0).lower() in allowed:
            return m.group(0)
        removed.append(m.group(0))
        return ''

    out = _LATIN_WORD.sub(repl, text)
    return tidy(out) if removed else out


# ---------------------------------------------------------------------------
# spacing / punctuation tidy-up (used after words were removed)
# ---------------------------------------------------------------------------

_EMPTY_BRACKETS = re.compile(r'[\(\[\{]\s*[,;:\-\u2013\s]*[\)\]\}]')
_SPACE_BEFORE   = re.compile(r'\s+([,.;:!?\u0964\u0965\)\]\}])')
_SPACE_AFTER    = re.compile(r'([\(\[\{])\s+')
_DOUBLE_PUNCT   = re.compile(r'([,;:])(\s*[,;:])+')


def tidy(text):
    prev = None
    while prev != text:                      # nested empty brackets "( ( ) )"
        prev = text
        text = _EMPTY_BRACKETS.sub('', text)
    text = _DOUBLE_PUNCT.sub(r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = _SPACE_BEFORE.sub(r'\1', text)
    text = _SPACE_AFTER.sub(r'\1', text)
    return text


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def finalize_output(hyp, src, tgt_lang, digits='native', latin_policy='source', recompose=True):
    """
    Raw system output  ->  final user-facing text in the target language.

    hyp          : decoded (and, in IndicTrans mode, IndicTrans-post-processed) text
    src          : the RAW source sentence (used by the Latin-word filter)
    digits       : 'native' (default) | 'ascii' | 'keep'
    latin_policy : 'source' (default) | 'keep' | 'drop'
    """
    text = normalize_text(strip_artifacts(hyp))
    text = filter_latin(text, src, latin_policy)
    if digits == 'native':
        text = localize_digits(ascii_digits_in_other_scripts(text, tgt_lang), tgt_lang)
    elif digits == 'ascii':
        text = ascii_digits(text)
    if recompose:
        text = recompose_nukta(text, tgt_lang)
    return text


def ascii_digits_in_other_scripts(text, lang):
    """A digit of a foreign script (e.g. Devanagari digits inside an Odia sentence) -> ASCII,
    so that localize_digits() can write it in the right script.  Native digits stay."""
    zero = LANGS[lang]['digit_zero']
    out = []
    for ch in text:
        o = ord(ch)
        if ch.isdigit() and not ch.isascii() and not (zero <= o <= zero + 9):
            out.append(str(int(ch)))
        else:
            out.append(ch)
    return ''.join(out)


def canonical(text):
    """Form used ONLY for scoring: artefacts removed, NFC, native digits -> ASCII,
    curly quotes -> straight.  Identical for every digit / nukta style."""
    return eval_normalize(normalize_text(strip_artifacts(text)))
