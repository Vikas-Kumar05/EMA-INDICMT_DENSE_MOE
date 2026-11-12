"""
train.py - Transformer Training Loop

Features
========
- Label-smoothed cross-entropy loss
- Gradient clipping + gradient accumulation
- ReduceLROnPlateau scheduler
- Early stopping
- BLEU evaluation via sacrebleu
- Checkpoint saving (best val-loss and best BLEU)
- Detailed CSV and text logs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import sacrebleu
import time
import math
import os
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from text_utils import LANG_TAGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_padding_mask(seq, pad_id=0):
    """[B, L] → [B, 1, 1, L]"""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def generate_causal_mask(size, device):
    """Lower-triangular boolean mask  [size, size]"""
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


def compute_perplexity(loss):
    if math.isnan(loss) or math.isinf(loss):
        return float('inf')
    return math.exp(min(loss, 100))


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LabelSmoothedCrossEntropy(nn.Module):
    """Label-smoothed cross-entropy with NaN guard"""

    def __init__(self, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing    = smoothing
        self.ignore_index = ignore_index
        self.confidence   = 1.0 - smoothing

    def forward(self, logits, target):
        B, L, V = logits.size()

        logits_flat = logits.float().reshape(-1, V)
        target_flat = target.reshape(-1)

        mask = target_flat != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_flat = torch.clamp(logits_flat, min=-100, max=100)
        log_probs   = F.log_softmax(logits_flat, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (V - 1))
            true_dist.scatter_(1, target_flat.unsqueeze(1), self.confidence)
            true_dist[:, self.ignore_index] = 0

        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-10)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer:
    """
    Full training loop for the pure Transformer model.

    Config keys (all optional, defaults shown)
    -------------------------------------------
    label_smoothing    : 0.1
    patience           : 5
    max_grad_norm      : 1.0
    grad_accum_steps   : 1
    val_every_n_epochs : 1
    save_dir           : 'checkpoints'
    """

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 device, config, tokenizer_path):

        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.config       = config

        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

        self.criterion = LabelSmoothedCrossEntropy(
            smoothing=config.get('label_smoothing', 0.1),
            ignore_index=0,
        )

        self.epoch       = 0
        self.global_step = 0

        self.best_val_loss   = float('inf')
        self.best_bleu       = 0.0
        self.patience        = config.get('patience', 5)
        self.patience_counter = 0
        self.early_stopped   = False

        self.max_grad_norm      = config.get('max_grad_norm', 1.0)
        self.grad_accum_steps   = config.get('grad_accum_steps', 1)
        self.val_every_n_epochs = config.get('val_every_n_epochs', 1)

        # bf16 autocast (GPU only), per-step vs per-epoch scheduler, extras
        self.use_amp      = bool(config.get('amp', False)) and torch.device(device).type == 'cuda'
        self.plateau      = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.model_config = getattr(model, 'config', {})
        self.eval_samples = config.get('eval_samples', 300)

        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.save_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

        self.train_log     = self.log_dir / 'training.log'
        self.val_log       = self.log_dir / 'validation.log'
        self.metrics_csv   = self.log_dir / 'metrics.csv'
        self.batch_csv     = self.log_dir / 'batch_metrics.csv'

        self._init_logs()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_logs(self):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.train_log, 'w') as f:
            f.write(f"Transformer Training Log\n")
            f.write(f"Early Stopping Patience: {self.patience}\n")
            f.write(f"Gradient Accumulation  : {self.grad_accum_steps} steps\n")
            f.write(f"Max Grad Norm          : {self.max_grad_norm}\n")
            f.write(f"Started: {ts}\n")
            f.write("=" * 70 + "\n\n")

        with open(self.val_log, 'w') as f:
            f.write(f"Transformer Validation Log\nStarted: {ts}\n")
            f.write("=" * 70 + "\n\n")

        with open(self.metrics_csv, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'train_ppl',
                'val_loss', 'val_ppl', 'bleu',
                'is_best_val', 'is_best_bleu', 'patience_counter',
                'lr', 'time_min', 'tokens_per_sec',
            ])

        with open(self.batch_csv, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'batch', 'global_step',
                'loss', 'ppl', 'lr', 'grad_norm',
                'num_tokens', 'time_sec',
            ])

        print(f"\n📁 Logs: {self.log_dir}")
        print(f"  Early Stopping Patience : {self.patience}")
        print(f"  Gradient Accumulation   : {self.grad_accum_steps} steps")
        print(f"  Max Gradient Norm       : {self.max_grad_norm}")

    def _log(self, log_file, message):
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()

        total_loss   = 0.0
        total_tokens = 0
        num_batches  = 0
        epoch_start  = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            src, tgt_in, tgt_out, src_mask, tgt_mask = [b.to(self.device) for b in batch]

            batch_start = time.time()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(src, tgt_in, src_mask, tgt_mask)
            loss = self.criterion(logits, tgt_out)

            # Scale for gradient accumulation
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            num_tokens  = (tgt_out != 0).sum().item()
            total_loss  += loss.item()
            total_tokens += num_tokens
            num_batches  += 1
            self.global_step += 1

            # Gradient step
            is_last = (batch_idx + 1) == len(self.train_loader)
            if (batch_idx + 1) % self.grad_accum_steps == 0 or is_last:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None and not self.plateau:
                    self.scheduler.step()          # warmup / inverse-sqrt: per optimizer step
            else:
                grad_norm = 0.0

            lr = self.optimizer.param_groups[0].get('lr', 0)

            # Batch CSV log (every 20 batches - opening the file every batch is slow)
            if (batch_idx + 1) % 20 == 0:
                gn = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                with open(self.batch_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        self.epoch, batch_idx + 1, self.global_step,
                        f"{loss.item():.6f}",
                        f"{compute_perplexity(loss.item()):.2f}",
                        lr, f"{gn:.4f}",
                        num_tokens,
                        f"{time.time() - batch_start:.3f}",
                    ])

            # Console progress
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - epoch_start
                tps     = total_tokens / elapsed if elapsed > 0 else 0
                print(
                    f"  Epoch {self.epoch} | Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"| Loss {loss.item():.4f} | PPL {compute_perplexity(loss.item()):.1f} "
                    f"| tok/s {tps:.0f}"
                )

        elapsed       = time.time() - epoch_start
        avg_loss      = total_loss / max(num_batches, 1)
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        msg = (
            f"Epoch {self.epoch} Train | loss={avg_loss:.4f} "
            f"ppl={compute_perplexity(avg_loss):.2f} "
            f"time={format_time(elapsed)} tok/s={tokens_per_sec:.0f}"
        )
        print(f"\n📊 {msg}")
        self._log(self.train_log, msg)

        return {
            'loss':           avg_loss,
            'perplexity':     compute_perplexity(avg_loss),
            'time':           elapsed,
            'tokens_per_sec': tokens_per_sec,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss  = 0.0
        num_batches = 0

        for batch in self.val_loader:
            src, tgt_in, tgt_out, src_mask, tgt_mask = [b.to(self.device) for b in batch]
            logits = self.model(src, tgt_in, src_mask, tgt_mask)
            loss   = self.criterion(logits, tgt_out)
            if not (math.isnan(loss.item()) or math.isinf(loss.item())):
                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        msg = (
            f"Epoch {self.epoch} Val   | loss={avg_loss:.4f} "
            f"ppl={compute_perplexity(avg_loss):.2f}"
        )
        print(f"✅ {msg}")
        self._log(self.val_log, msg)

        return {'loss': avg_loss, 'perplexity': compute_perplexity(avg_loss)}

    # ------------------------------------------------------------------
    # BLEU evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_translation(self, num_samples=None):
        """
        Greedy-decode an evenly spaced sample of the dev set and score each
        translation DIRECTION separately (identified by the <2xxx> tag that
        follows <BOS> in the source). The old version took the first 100 rows,
        which is only one direction of one language pair.
        """
        self.model.eval()
        num_samples = num_samples or self.eval_samples
        ds = self.val_loader.dataset
        n  = len(ds)
        k  = min(num_samples, n)
        idxs = sorted(set(int(i * n / k) for i in range(k)))

        tag_of = {self.tokenizer.piece_to_id(t): t for t in LANG_TAGS}
        hyps, refs = defaultdict(list), defaultdict(list)

        for i in idxs:
            src, tgt = ds[i]
            direction = tag_of.get(int(src[1]), '<?>') if len(src) > 1 else '<?>'
            try:
                pred = self.greedy_decode(src.unsqueeze(0).to(self.device),
                                          max_len=int(len(src) * 2) + 10)
                hyps[direction].append(self.tokenizer.decode(pred.tolist()))
                refs[direction].append(self.tokenizer.decode(tgt.tolist()))
            except Exception as e:
                print(f"   ⚠️ decode failed for sample {i}: {e}")

        if not hyps:
            return {'bleu': 0.0, 'chrf': 0.0, 'per_direction': {}}

        per_dir, bleus, chrfs = {}, [], []
        for d in sorted(hyps):
            b = sacrebleu.corpus_bleu(hyps[d], [refs[d]]).score
            c = sacrebleu.corpus_chrf(hyps[d], [refs[d]]).score
            per_dir[d] = (b, c, len(hyps[d]))
            bleus.append(b); chrfs.append(c)
            print(f"   ->{d[2:-1]:<4} BLEU {b:6.2f} | chrF {c:6.2f} | n={len(hyps[d])}")

        msg = (f"Epoch {self.epoch} Greedy | BLEU(avg)={sum(bleus)/len(bleus):.2f} "
               f"chrF(avg)={sum(chrfs)/len(chrfs):.2f}  "
               + "  ".join(f"{d}:{v[0]:.1f}" for d, v in per_dir.items()))
        self._log(self.val_log, msg)
        return {'bleu': sum(bleus) / len(bleus), 'chrf': sum(chrfs) / len(chrfs),
                'per_direction': per_dir}

    def greedy_decode(self, src, max_len=200, bos_id=2, eos_id=3):
        # never exceed the positional-encoding table (was a silent crash before)
        max_pos = self.model.embedding.pos_encoder.pe.size(1)
        max_len = min(max_len, max_pos - 1)

        src_mask       = generate_padding_mask(src).to(self.device)
        encoder_output = self.model.encode(src, src_mask)
        tgt            = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=self.device)

        for _ in range(max_len):
            logits     = self.model.decode_step(tgt, encoder_output, src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt        = torch.cat([tgt, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return tgt[0]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_best_val=False, is_best_bleu=False, suffix=''):
        ckpt = {
            'epoch':                self.epoch,
            'global_step':          self.global_step,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss':        self.best_val_loss,
            'best_bleu':            self.best_bleu,
            'patience_counter':     self.patience_counter,
            'config':               self.config,
            'model_config':         self.model_config,
        }
        if self.scheduler is not None:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.save_dir / f'checkpoint_epoch_{self.epoch}{suffix}.pt'
        torch.save(ckpt, path)

        if is_best_val:
            torch.save(ckpt, self.save_dir / 'best_model.pt')
            print(f"🏆 Best validation model saved.")

        if is_best_bleu:
            torch.save(ckpt, self.save_dir / 'best_bleu_model.pt')
            print(f"📈 Best BLEU model saved.")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs):
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING")
        print(f"  Early stopping patience: {self.patience}")
        print(f"{'='*60}\n")

        start_epoch = self.epoch + 1

        for epoch in range(start_epoch, num_epochs + 1):
            self.epoch = epoch

            train_metrics = self.train_epoch()

            if math.isnan(train_metrics['loss']):
                print("❌ NaN detected — stopping training.")
                break

            should_validate = (
                epoch % self.val_every_n_epochs == 0
                or epoch == num_epochs
                or epoch == start_epoch
            )

            if should_validate:
                val_metrics   = self.validate()
                trans_metrics = self.evaluate_translation()

                if self.scheduler is not None and self.plateau:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_metrics['loss'])
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr < old_lr:
                        print(f"📉 LR reduced: {old_lr:.6f} → {new_lr:.6f}")

                is_best_val  = val_metrics['loss'] < self.best_val_loss
                is_best_bleu = trans_metrics['bleu'] > self.best_bleu

                if is_best_val:
                    self.best_val_loss    = val_metrics['loss']
                    self.patience_counter = 0
                    print(f"✅ New best val loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"⚠️  No improvement. Patience: {self.patience_counter}/{self.patience}")

                if is_best_bleu:
                    self.best_bleu = trans_metrics['bleu']
                    print(f"🌟 New best BLEU: {self.best_bleu:.2f}")

            else:
                val_metrics   = {'loss': float('nan'), 'perplexity': float('inf')}
                trans_metrics = {'bleu': 0.0}
                is_best_val   = is_best_bleu = False
                next_val = (epoch // self.val_every_n_epochs + 1) * self.val_every_n_epochs
                print(f"⏭️  Skipping validation (next at epoch {next_val})")

            lr = self.optimizer.param_groups[0].get('lr', -1)

            with open(self.metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['perplexity']:.2f}",
                    f"{val_metrics['loss']:.6f}"   if not math.isnan(val_metrics['loss'])   else "N/A",
                    f"{val_metrics['perplexity']:.2f}" if not math.isinf(val_metrics['perplexity']) else "N/A",
                    f"{trans_metrics['bleu']:.2f}",
                    int(is_best_val),
                    int(is_best_bleu),
                    self.patience_counter,
                    lr,
                    f"{train_metrics['time']/60:.2f}",
                    f"{train_metrics['tokens_per_sec']:.1f}",
                ])

            if epoch % 5 == 0 or is_best_val or is_best_bleu:
                self.save_checkpoint(is_best_val=is_best_val, is_best_bleu=is_best_bleu)

            if should_validate and self.patience_counter >= self.patience:
                print(
                    f"\n⏹️  Early stopping at epoch {self.epoch}. "
                    f"Best val loss: {self.best_val_loss:.4f}  Best BLEU: {self.best_bleu:.2f}"
                )
                self.save_checkpoint(suffix='_early_stop')
                self.early_stopped = True
                break

        print(
            f"\n{'='*60}\n"
            f"TRAINING COMPLETE\n"
            f"  Best val loss : {self.best_val_loss:.4f}\n"
            f"  Best BLEU     : {self.best_bleu:.2f}\n"
            f"{'='*60}\n"
        )
