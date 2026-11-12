"""
indic_utils.py - IndicTrans toolkit pre-/post-processing for the Indic MT pipeline


"""

import glob
import importlib.machinery
import importlib.util
import os

from text_utils import LANGS

INSTALL_HINT = ("IndicTransToolkit is required for --indictrans.\n"
                "   pip install IndicTransToolkit\n"
                "(it needs a C compiler / Cython to build; see README 'Troubleshooting')")


def _load_processor_class():
    try:
        from IndicTransToolkit import IndicProcessor
        return IndicProcessor
    except ImportError as first_err:
        spec = importlib.util.find_spec("IndicTransToolkit")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError(INSTALL_HINT) from first_err
        pkg_dir = list(spec.submodule_search_locations)[0]
        for pat in ("processor*.so", "processor*.pyd"):
            for path in glob.glob(os.path.join(pkg_dir, pat)):
                try:
                    loader = importlib.machinery.ExtensionFileLoader("processor", path)
                    mod_spec = importlib.util.spec_from_file_location("processor", path, loader=loader)
                    mod = importlib.util.module_from_spec(mod_spec)
                    loader.exec_module(mod)
                    return mod.IndicProcessor
                except Exception:
                    continue
        raise ImportError(INSTALL_HINT + f"\n(original error: {first_err})") from first_err


def indictrans_available():
    try:
        _load_processor_class()
        return True
    except ImportError:
        return False


class IndicTransPreprocessor:
    """
    Thin, safe wrapper around IndicProcessor working with this project's language codes
    ('asm', 'brx', 'ben', 'odi').

    placeholders=False (default)  numerals / URLs stay as ordinary text (digits are still
                                  mapped to ASCII).  Best for a from-scratch model.
    placeholders=True             URLs / e-mails / long numerals become <IDn> tokens that are
                                  copied back from the source sentence after decoding.
    """

    def __init__(self, placeholders=False):
        cls = _load_processor_class()
        self.placeholders = bool(placeholders)
        # `inference=True` stores placeholder maps in the queue; `inference=False` does not.
        self._pre  = cls(inference=self.placeholders)
        # the post-processing instance always runs with inference=True so its queue exists;
        # when placeholders are off we push empty maps ourselves (see _push_empty_maps)
        self._post = self._pre if self.placeholders else cls(inference=True)

    # ------------------------------------------------------------------
    @staticmethod
    def flores(lang):
        return LANGS[lang]['flores']

    def _push_empty_maps(self, n, lang):
        # '.' contains no entity -> the processor queues an empty placeholder map
        self._post.preprocess_batch(['.'] * n, src_lang=self.flores(lang), is_target=True)

    def _drain(self, n, lang):
        """Pop n queued maps (used after bulk preprocessing)."""
        self._post.postprocess_batch([''] * n, lang=self.flores(lang))

    # ------------------------------------------------------------------
    # corpus-scale preprocessing (training data) - no postprocess needed
    # ------------------------------------------------------------------
    def preprocess_corpus(self, texts, lang, chunk=2000, verbose=True):
        out, n = [], len(texts)
        for i in range(0, n, chunk):
            part = [str(t) for t in texts[i:i + chunk]]
            res = self._pre.preprocess_batch(part, src_lang=self.flores(lang), is_target=True)
            if self.placeholders:
                self._drain(len(part), lang)          # keep the internal queue empty
            out.extend(res)
            if verbose and ((i // chunk) % 50 == 49 or i + chunk >= n):
                print(f"      indictrans[{lang}] {min(i + chunk, n):,}/{n:,}")
        return out

    # ------------------------------------------------------------------
    # sentence-level pipeline (inference)
    # ------------------------------------------------------------------
    def preprocess_one(self, text, lang):
        """Preprocess ONE source sentence; MUST be followed by exactly one postprocess_one."""
        res = self._pre.preprocess_batch([text], src_lang=self.flores(lang), is_target=True)[0]
        if not self.placeholders:
            self._push_empty_maps(1, lang)
        return res

    def postprocess_one(self, text, lang):
        return self._post.postprocess_batch([text], lang=self.flores(lang))[0]

    def translate_one(self, text, src_lang, tgt_lang, translate_fn):
        """
        raw source -> preprocess -> translate_fn(model_space_source) -> postprocess.
        The queue stays in sync even if translate_fn raises.
        """
        model_src = self.preprocess_one(text, src_lang)
        try:
            hyp = translate_fn(model_src)
        except Exception:
            hyp = ""
        return self.postprocess_one(hyp, tgt_lang)
