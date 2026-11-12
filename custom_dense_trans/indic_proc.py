"""
indic_proc.py - IndicTransToolkit pre- / post-processing wrapper.

What the toolkit's IndicProcessor does (verified on the installed version)
=========================================================================
preprocess : punctuation normalisation (curly quotes, "…" -> "...", ...),
             Indic digits -> ASCII digits, Indic-NLP unicode normalisation
             (nukta etc.), Indic tokenisation (punctuation split from words),
             transliteration of the language's script into Devanagari so
             related languages share one script / one vocabulary.
postprocess: transliteration back to the native script, Odia fixes,
             Indic detokenisation.

Why a wrapper
=============
* ``import IndicTransToolkit`` runs the package __init__, which imports a
  HF-transformers collator that is broken on some transformers versions.
  We only need the compiled processor, so we fall back to loading it directly.
* postprocess_batch() pulls one "placeholder map" per sentence from an internal
  queue that only preprocess_batch(inference=True) fills, and it BLOCKS FOREVER
  when the queue is empty. ``from_model`` feeds the queue with empty maps first.
* The toolkit prepends "<src> <tgt>" language tags. We never want them (this
  project has its own <2xxx> tags), so targets are processed with is_target=True.
* Entity placeholders (<ID1> for numbers / URLs) are NOT used: the model is
  trained from scratch and never sees them.

Modes
=====
off      identity - model sees only the generic normalisation of text_utils
toolkit  to_model / from_model go through IndicProcessor
"""

import glob
import importlib.machinery
import importlib.util
import os
import sys

# project language code -> FLORES code used by IndicTransToolkit
FLORES = {'asm': 'asm_Beng', 'brx': 'brx_Deva', 'ben': 'ben_Beng', 'odi': 'ory_Orya'}
# project language code -> ISO code used by indic-nlp-library
ISO = {'asm': 'as', 'brx': 'hi', 'ben': 'bn', 'odi': 'or'}

MODES = ('off', 'toolkit')


def _load_processor_class():
    """Return IndicProcessor class or None if the toolkit is not usable."""
    try:
        from IndicTransToolkit.processor import IndicProcessor
        return IndicProcessor
    except Exception:
        pass
    # Package __init__ failed (usually the transformers collator): load the
    # compiled / pure-python processor module straight from its folder.
    try:
        spec = importlib.util.find_spec('IndicTransToolkit')   # does not execute __init__
        if spec is None or not spec.submodule_search_locations:
            return None
        folder = list(spec.submodule_search_locations)[0]
        cands = []
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            cands += glob.glob(os.path.join(folder, 'processor*' + suffix.split('.')[-1]))
        cands += glob.glob(os.path.join(folder, 'processor.py'))
        for path in cands:
            s = importlib.util.spec_from_file_location('processor', path)
            mod = importlib.util.module_from_spec(s)
            s.loader.exec_module(mod)
            return mod.IndicProcessor
    except Exception:
        return None
    return None


def toolkit_available():
    return _load_processor_class() is not None


class IndicText:
    """Convert text between 'native script' and 'model space'."""

    def __init__(self, mode='toolkit'):
        if mode not in MODES:
            raise ValueError(f"indic mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self._pre = self._post = None
        if mode == 'toolkit':
            cls = _load_processor_class()
            if cls is None:
                raise RuntimeError(
                    "indic mode 'toolkit' needs IndicTransToolkit "
                    "(pip install IndicTransToolkit indic-nlp-library) - or use --indic-mode off")
            self._pre = cls(inference=False)    # no placeholders
            self._post = cls(inference=True)    # owns the queue postprocess_batch reads

    # ------------------------------------------------------------------
    def to_model(self, texts, lang):
        """native-script sentences -> model-space sentences (no language tags)."""
        texts = list(texts)
        if self.mode == 'off' or not texts:
            return texts
        out = self._pre.preprocess_batch(texts, src_lang=FLORES[lang],
                                         tgt_lang=FLORES[lang], is_target=True)
        return [o.strip() for o in out]

    def from_model(self, texts, lang):
        """model-space sentences -> native-script sentences."""
        texts = list(texts)
        if self.mode == 'off' or not texts:
            return texts
        # one (empty) placeholder map per sentence, otherwise postprocess_batch blocks
        self._post.preprocess_batch(['a'] * len(texts), src_lang='hin_Deva', tgt_lang='hin_Deva')
        return [o.strip() for o in self._post.postprocess_batch(texts, lang=FLORES[lang])]

    def canonical(self, texts, lang):
        """
        The form a reference is compared in: from_model(to_model(ref)).
        Predictions are produced by from_model(...), so both sides carry the same
        toolkit conventions (ASCII digits, "..." for ellipsis, detokenised spacing)
        and no metric point is lost on pure formatting differences.
        """
        if self.mode == 'off':
            return list(texts)
        return self.from_model(self.to_model(texts, lang), lang)
