"""
prepare_dataset.py - Dataset loading, caching and batching

Changes vs. the original
========================
* Cache file name now includes a hash of the CSV *and* the tokenizer, so
  switching datasets (asm-bodo -> bengali-odia) can never load a stale cache.
* Over-long pairs are DROPPED instead of truncated. Truncation used to cut off
  the <EOS> token and taught the model misaligned / never-ending outputs.
* Length-bucketed batches (similar-length sentences together) -> far less
  padding, noticeably faster epochs.
"""

import os
import math
import hashlib
from pathlib import Path

import torch
import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


def _file_md5(path, n_bytes=None):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read(n_bytes) if n_bytes else f.read())
    return h.hexdigest()[:8]


class TranslationDataset(Dataset):
    """Pre-tokenised (src_ids, tgt_ids) pairs with on-disk caching."""

    def __init__(self, csv_file, sp_model, src_col='src', tgt_col='tgt',
                 max_len=128, cache_dir='cache', drop_long=True):
        self.max_len   = max_len
        self.drop_long = drop_long
        self.dropped   = 0

        os.makedirs(cache_dir, exist_ok=True)
        name = (f"{Path(csv_file).stem}_len{max_len}_{int(drop_long)}"
                f"_{_file_md5(csv_file)}_{_file_md5(sp_model)}.pt")
        self.cache_file = os.path.join(cache_dir, name)

        if os.path.exists(self.cache_file):
            print(f"📦 Loading cached dataset: {self.cache_file}")
            self._load_cache()
        else:
            print(f"🔨 Pre-tokenizing dataset: {csv_file}")
            self._tokenize_and_cache(csv_file, sp_model, src_col, tgt_col)

        self.lengths = [max(len(s), len(t)) for s, t in self.encoded_data]
        print(f"✅ Dataset ready: {len(self.encoded_data):,} samples "
              f"(dropped {self.dropped:,} too long/invalid)")

    def _load_cache(self):
        try:
            blob = torch.load(self.cache_file, weights_only=False)
        except TypeError:                      # older torch without weights_only
            blob = torch.load(self.cache_file)
        self.encoded_data = blob['data']
        self.dropped      = blob.get('dropped', 0)

    def _tokenize_and_cache(self, csv_file, sp_model, src_col, tgt_col):
        df = pd.read_csv(csv_file, dtype=str).dropna(subset=[src_col, tgt_col])
        print(f"   Read {len(df):,} rows")
        sp = spm.SentencePieceProcessor(model_file=sp_model)

        srcs = df[src_col].str.strip().tolist()
        tgts = df[tgt_col].str.strip().tolist()

        self.encoded_data = []
        chunk = 20000
        for i in range(0, len(srcs), chunk):
            s_ids = sp.encode(srcs[i:i + chunk], out_type=int, add_bos=True, add_eos=True)
            t_ids = sp.encode(tgts[i:i + chunk], out_type=int, add_bos=True, add_eos=True)
            for s, t in zip(s_ids, t_ids):
                if len(s) < 3 or len(t) < 3:                 # BOS + EOS + >=1 token
                    self.dropped += 1
                    continue
                if len(s) > self.max_len or len(t) > self.max_len:
                    if self.drop_long:
                        self.dropped += 1
                        continue
                    s = s[:self.max_len - 1] + [sp.eos_id()]
                    t = t[:self.max_len - 1] + [sp.eos_id()]
                self.encoded_data.append((torch.tensor(s, dtype=torch.long),
                                          torch.tensor(t, dtype=torch.long)))
            print(f"   Tokenized {min(i + chunk, len(srcs)):,} / {len(srcs):,}")

        if not self.encoded_data:
            raise ValueError("No valid samples after tokenization!")
        if self.dropped > 0.05 * len(srcs):
            print(f"   ⚠️ {self.dropped:,} pairs exceeded max_len={self.max_len} - "
                  f"consider a larger --max-len")

        torch.save({'data': self.encoded_data, 'dropped': self.dropped}, self.cache_file)
        print(f"   💾 Cached to: {self.cache_file}")

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


class BucketBatchSampler(Sampler):
    """Shuffle, cut into big chunks, sort each chunk by length, batch, shuffle batches."""

    def __init__(self, lengths, batch_size, shuffle=True, chunk_mult=50):
        self.lengths    = lengths
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.chunk      = batch_size * chunk_mult

    def __iter__(self):
        n = len(self.lengths)
        idx = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        batches = []
        for i in range(0, n, self.chunk):
            part = sorted(idx[i:i + self.chunk], key=lambda j: self.lengths[j])
            batches += [part[k:k + self.batch_size] for k in range(0, len(part), self.batch_size)]
        if self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[o] for o in order]
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


def collate_fn(batch, pad_id=0):
    """
    Returns src, tgt_in, tgt_out, src_mask [B,1,1,S], tgt_mask [B,1,T,T].
    Mask convention: True = may attend, False = masked.
    """
    src_seqs, tgt_seqs = zip(*batch)
    src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=pad_id)
    tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_id)

    tgt_input  = tgt_batch[:, :-1]      # <BOS> w1 w2 ...
    tgt_output = tgt_batch[:, 1:]       # w1 w2 ... <EOS>

    src_mask = (src_batch != pad_id).unsqueeze(1).unsqueeze(2)

    T = tgt_input.size(1)
    tgt_pad = (tgt_input != pad_id).unsqueeze(1).unsqueeze(2).expand(-1, -1, T, -1)
    causal  = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    return src_batch, tgt_input, tgt_output, src_mask, tgt_pad & causal


def create_dataloader(csv_file, sp_model, src_col='src', tgt_col='tgt',
                      batch_size=64, max_len=128, shuffle=True, num_workers=2,
                      cache_dir='cache', drop_long=True):
    dataset = TranslationDataset(csv_file, sp_model, src_col, tgt_col,
                                 max_len=max_len, cache_dir=cache_dir, drop_long=drop_long)
    sampler = BucketBatchSampler(dataset.lengths, batch_size, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    print(f"📊 DataLoader: {len(dataset):,} samples | {len(loader)} batches | batch size {batch_size}")
    return loader
