"""
inference.py - Model loading, beam search and the full translation pipeline


"""

import os
import re

import torch

from model import TransformerModel
from text_utils import LANGS, LANG_TAGS, normalize_text, strip_tags


# ============================================================================
# CONFIG
# ============================================================================

class DecodeConfig:
    # pipeline flags (filled from the checkpoint)
    use_indictrans     = False
    indic_placeholders = False

    # beam search
    beam_size          = 5
    length_ratio       = 2.0      # dynamic max_len = src_len * length_ratio
    base_max_len       = 50       # ... but at least this many tokens
    hard_max_len       = 128      # ... and at most this many (training max_len)
    repetition_penalty = 1.0      # 1.0 = off (1.5 hurts legitimate repeats in Indic text)
    length_penalty     = 0.6      # GNMT alpha
    temperature        = 1.0
    min_decode_len     = 3
    unk_penalty        = 0.0      # log-prob penalty on <unk> when copy-back is possible
    early_stopping     = True     # stop as soon as `beam_size` hypotheses are finished

    # output post-processing (postprocess.finalize_output)
    digits             = 'native'   # 'native' | 'ascii' | 'keep'
    latin_policy       = 'source'   # 'source' | 'keep' | 'drop'
    recompose_nukta    = True

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path, device):
    """Load TransformerModel from a checkpoint saved by train.py.
    Returns (model, train_config) - train_config holds data_name / pairs / IndicTrans / MoE flags."""
    print(f"\n{'='*60}\nLOADING MODEL\n{'='*60}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:                                   # older torch without weights_only
        checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg = checkpoint.get('model_config') or checkpoint.get('config', {})
    if 'vocab_size' not in cfg:
        raise ValueError("Checkpoint has no model_config - it was saved by the old train.py. "
                         "Re-train with the updated code.")
    model = TransformerModel(
        vocab_size         = cfg['vocab_size'],
        d_model            = cfg.get('d_model', 512),
        nhead              = cfg.get('nhead', 8),
        d_ff               = cfg.get('d_ff', 1024),
        num_encoder_layers = cfg.get('num_encoder_layers', 6),
        num_decoder_layers = cfg.get('num_decoder_layers', 6),
        dropout            = cfg.get('dropout', 0.1),
        max_seq_len        = cfg.get('max_seq_len', 256),
        moe_every          = cfg.get('moe_every', 0),          # absent in baseline checkpoints
        moe_num_experts    = cfg.get('moe_num_experts', 4),
        moe_top_k          = cfg.get('moe_top_k', 2),
        moe_jitter         = cfg.get('moe_jitter', 0.0),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device).eval()

    print(f"  ✅ Loaded from epoch {checkpoint.get('epoch', 'unknown')}  "
          f"(best BLEU: {checkpoint.get('best_bleu', 'unknown')})")
    print(f"  Parameters: {model.get_num_params():,}  |  MoE layers: {len(model.moe_layers())}")
    return model, checkpoint.get('config', {})


# ============================================================================
# MASKS
# ============================================================================

def _padding_mask(seq, pad_id=0):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)            # [B,1,1,L] True = real token


# ============================================================================
# BEAM SEARCH
# ============================================================================

class BeamSearchDecoder:
    """Batched beam search for the Transformer (works for dense and MoE models)."""

    def __init__(self, model: TransformerModel, tokenizer, config: DecodeConfig):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device = config.device
        # IndicTrans mode: text is already in "model space" (tokenised / transliterated);
        # generic re-normalisation would undo the toolkit's spacing, so it is skipped.
        self.model_space = bool(config.use_indictrans)

        self.bos_id, self.eos_id = tokenizer.bos_id(), tokenizer.eos_id()
        self.pad_id, self.unk_id = tokenizer.pad_id(), tokenizer.unk_id()
        self.tag_ids  = [tokenizer.piece_to_id(t) for t in LANG_TAGS]
        self.max_pos  = model.embedding.pos_encoder.pe.size(1)
        self._oov_cache = {}

    # ------------------------------------------------------------------
    # out-of-vocabulary source characters (for <unk> copy-back)
    # ------------------------------------------------------------------
    def _is_oov(self, ch):
        hit = self._oov_cache.get(ch)
        if hit is None:
            hit = self.unk_id in self.tokenizer.encode(ch)
            self._oov_cache[ch] = hit
        return hit

    def oov_chars(self, text):
        """Characters of `text` the tokenizer cannot represent, in order of appearance."""
        text = strip_tags(text)
        return [c for c in text if not c.isspace() and c.isprintable() and self._is_oov(c)]

    # ------------------------------------------------------------------
    # ids -> text, putting source characters back where the model produced <unk>
    # ------------------------------------------------------------------
    def ids_to_text(self, ids, oov):
        tok, parts, buf = self.tokenizer, [], []
        replacements = iter(oov)

        def flush():
            if not buf:
                return
            piece = tok.decode(buf)
            if parts and tok.id_to_piece(buf[0]).startswith('\u2581'):
                piece = ' ' + piece                  # decode() strips the leading word mark
            parts.append(piece)
            buf.clear()

        for t in ids:
            if t == self.unk_id:
                flush()
                parts.append(next(replacements, ''))
            elif t not in (self.bos_id, self.eos_id, self.pad_id):
                buf.append(t)
        flush()
        text = ''.join(parts).replace('\u2581', ' ').replace('<unk>', '')
        text = strip_tags(text)
        return re.sub(r'\s+', ' ', text).strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _lp(length, alpha):
        return ((5.0 + length) / 6.0) ** alpha

    @torch.no_grad()
    def decode(self, src_text, beam_size=None, max_len=None):
        cfg  = self.config
        k    = beam_size or cfg.beam_size
        text = src_text if self.model_space else normalize_text(src_text)
        oov  = self.oov_chars(text)

        src_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(src_ids) > self.max_pos - 2:                    # never exceed the positional table
            src_ids = src_ids[:self.max_pos - 3] + [self.eos_id]
        src_len = len(src_ids)

        if max_len is None:
            max_len = max(cfg.base_max_len, min(int(src_len * cfg.length_ratio), cfg.hard_max_len))
        max_len = min(max_len, self.max_pos - 2)
        min_len = max(cfg.min_decode_len, int(src_len * 0.3))
        alpha   = 0.4 if src_len > 100 else cfg.length_penalty

        src      = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_mask = _padding_mask(src, self.pad_id)
        enc      = self.model.encode(src, src_mask)
        if torch.isnan(enc).any():
            return ""
        enc, src_mask = enc.expand(k, -1, -1), src_mask.expand(k, -1, -1, -1)

        seqs   = torch.full((1, 1), self.bos_id, dtype=torch.long, device=self.device)
        scores = torch.zeros(1, device=self.device)
        finished = []                                          # (normalised score, token list)

        for _ in range(max_len):
            n, cur_len = seqs.size(0), seqs.size(1)            # cur_len = BOS + generated tokens
            logits = self.model.decode_step(seqs, enc[:n], src_mask[:n])[:, -1, :].float()
            if torch.isnan(logits).any():
                break
            logits = logits / cfg.temperature

            if cfg.repetition_penalty != 1.0:
                for b in range(n):
                    for tok in set(seqs[b].tolist()) - {self.bos_id}:
                        v = logits[b, tok]
                        logits[b, tok] = v / cfg.repetition_penalty if v > 0 else v * cfg.repetition_penalty

            neg_inf = float('-inf')
            logits[:, self.pad_id] = neg_inf
            logits[:, self.bos_id] = neg_inf
            logits[:, self.tag_ids] = neg_inf
            if oov:                                            # copy-back possible: allow <unk>
                logits[:, self.unk_id] -= cfg.unk_penalty
            else:                                              # nothing to copy: never emit <unk>
                logits[:, self.unk_id] = neg_inf
            if cur_len - 1 < min_len:
                logits[:, self.eos_id] = neg_inf

            cand = scores[:, None] + torch.log_softmax(logits, dim=-1)       # [n, V]
            V = cand.size(1)
            top_s, top_i = cand.view(-1).topk(min(2 * k, cand.numel()))

            keep_rows, keep_toks, keep_scores = [], [], []
            for rank, (s, i) in enumerate(zip(top_s.tolist(), top_i.tolist())):
                row, tok = divmod(i, V)
                if s == neg_inf:
                    continue
                if tok == self.eos_id:
                    if rank < k:                               # EOS only counts inside the top-k
                        finished.append((s / self._lp(cur_len, alpha), seqs[row].tolist() + [tok]))
                    continue
                keep_rows.append(row); keep_toks.append(tok); keep_scores.append(s)
                if len(keep_rows) == k:
                    break

            if not keep_rows:
                break
            if cfg.early_stopping and len(finished) >= k:
                break

            seqs   = torch.cat([seqs[keep_rows],
                                torch.tensor(keep_toks, dtype=torch.long, device=self.device)[:, None]], 1)
            scores = torch.tensor(keep_scores, device=self.device)

        # hypotheses still running when max_len is reached
        for row in range(seqs.size(0)):
            if seqs[row, -1].item() != self.eos_id and seqs.size(1) > 1:
                finished.append((scores[row].item() / self._lp(seqs.size(1), alpha),
                                 seqs[row].tolist() + [self.eos_id]))

        if not finished:
            return ""
        best = max(finished, key=lambda x: x[0])[1]
        return self.ids_to_text(best, oov)


# ============================================================================
# FULL TRANSLATION PIPELINE
# ============================================================================

class Translator:
    """raw sentence in source language -> final sentence in target language."""

    def __init__(self, model, tokenizer, config: DecodeConfig, indic=None):
        self.config  = config
        self.indic   = indic
        self.decoder = BeamSearchDecoder(model, tokenizer, config)

    def translate_raw(self, raw_src, src_lang, tgt_lang):
        """Decoded text BEFORE digit / English / nukta post-processing."""
        tag = LANGS[tgt_lang]['tag']
        raw_src = raw_src.strip()
        if self.indic is None:
            return self.decoder.decode(f"{tag} {raw_src}")
        return self.indic.translate_one(
            raw_src, src_lang, tgt_lang, lambda model_src: self.decoder.decode(f"{tag} {model_src}"))

    def finalize(self, hyp, raw_src, tgt_lang):
        from postprocess import finalize_output
        return finalize_output(hyp, raw_src, tgt_lang, digits=self.config.digits,
                               latin_policy=self.config.latin_policy,
                               recompose=self.config.recompose_nukta)

    def translate(self, raw_src, src_lang, tgt_lang):
        return self.finalize(self.translate_raw(raw_src, src_lang, tgt_lang), raw_src, tgt_lang)
