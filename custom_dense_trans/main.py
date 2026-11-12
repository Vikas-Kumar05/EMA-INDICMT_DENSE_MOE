"""
main.py - Train a Transformer for  Assamese<->Bodo  and/or  Bengali<->Odia

Data files (in --data-dir, header row required, extra columns ignored):
    asm_bodo.csv|xlsx       columns: asm, bodo
    bengali_odia.csv|xlsx   columns: bengali, odia
Optional pre-made splits (otherwise 1000 dev / 1000 test are split off automatically):
    asm_bodo_dev.csv / asm_bodo_test.csv,  bengali_odia_dev.csv / bengali_odia_test.csv

Examples
--------
python main.py                                  # both pairs, ONE multilingual model (4 directions)
python main.py --pairs asm-brx                  # Assamese<->Bodo only
python main.py --pairs ben-odi --epochs 60
python main.py --asm-bodo-file /path/to/file.xlsx --data-dir /path/to/data
"""

import os
import json
import math
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from text_utils import PAIRS
from indic_proc import toolkit_available
from text_utils import PREPROC_VERSION
from tokeniser import prepare_all, train_sentencepiece, verify_tokenizer, report_unk_rate
from prepare_dataset import create_dataloader
from model import TransformerModel
from train import TransformerTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train Transformer MT (asm-brx / ben-odi)')

    # data
    p.add_argument('--pairs', nargs='+', choices=list(PAIRS), default=list(PAIRS))
    p.add_argument('--data-dir', default='data')
    p.add_argument('--asm-bodo-file', default=None)
    p.add_argument('--ben-odi-file', default=None)
    p.add_argument('--asm-bodo-dev', default=None)
    p.add_argument('--asm-bodo-test', default=None)
    p.add_argument('--ben-odi-dev', default=None)
    p.add_argument('--ben-odi-test', default=None)
    p.add_argument('--run-name', default=None, help='default: pairs joined with "+"')
    p.add_argument('--dev-size', type=int, default=1000)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--vocab-size', type=int, default=16000)
    p.add_argument('--max-len', type=int, default=128, help='max tokens per training sentence')
    p.add_argument('--max-pos', type=int, default=256, help='positional table size (>= max-len+2)')
    p.add_argument('--force-reprocess', action='store_true')
    p.add_argument('--indic-mode', choices=['auto', 'toolkit', 'off'], default='auto',
                   help="IndicTransToolkit pre/post-processing: normalise + tokenise + transliterate "
                        "to a shared Devanagari space for training, and back to native script "
                        "after decoding. auto = toolkit if installed, else off")
    p.add_argument('--dedupe-sides', action='store_true',
                   help='keep one row per distinct source and per distinct target sentence')
    p.add_argument('--max-words', type=int, default=150, help='drop training rows longer than this')
    p.add_argument('--num-workers', type=int, default=2)

    # model
    p.add_argument('--d-model', type=int, default=512)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--enc-layers', type=int, default=6)
    p.add_argument('--dec-layers', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.1)

    # optimisation
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--scheduler', choices=['warmup', 'plateau'], default='warmup')
    p.add_argument('--warmup-steps', type=int, default=2000)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--val-every-n-epochs', type=int, default=1)
    p.add_argument('--amp', action='store_true', help='bf16 autocast on GPU')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------

def initialize_weights(model):
    """
    Attention and embeddings already initialise themselves. Here we only give the
    SwiGLU feed-forward layers a Xavier init.

    The previous version re-initialised EVERY nn.Linear - including the output
    projection, whose weight is TIED to the embedding matrix. That overwrote the
    embeddings with a much smaller scale and un-zeroed the padding row.
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and '.ffn.' in name:
            nn.init.xavier_uniform_(m.weight)
    with torch.no_grad():
        model.embedding.subword_embed.weight[0].fill_(0)     # keep <pad> row at zero
    assert model.output_proj.weight is model.embedding.subword_embed.weight, "weight tying lost"


def test_forward_pass(model, device, vocab_size):
    model.eval()
    B, S, T = 2, 20, 15
    src = torch.randint(4, vocab_size, (B, S), device=device); src[:, 15:] = 0
    tgt = torch.randint(4, vocab_size, (B, T), device=device); tgt[:, 10:] = 0
    with torch.no_grad():
        logits = model(src, tgt)
    assert logits.shape == (B, T, vocab_size)
    assert not torch.isnan(logits).any()
    print(f"🧪 Forward pass OK  logits {tuple(logits.shape)}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def setup_data(args, run_name):
    proc_dir = os.path.join('processed', run_name)
    spm_prefix = os.path.join('tokenizer', run_name, 'spm')
    spm_model = spm_prefix + '.model'
    train_csv = os.path.join(proc_dir, 'bidir_train.csv')
    dev_csv = os.path.join(proc_dir, 'bidir_dev.csv')

    # resolve preprocessing mode; changing it invalidates processed data + tokenizer
    mode = args.indic_mode
    if mode == 'auto':
        mode = 'toolkit' if toolkit_available() else 'off'
        if mode == 'off':
            print("⚠️  IndicTransToolkit not usable - continuing with --indic-mode off "
                  "(pip install IndicTransToolkit indic-nlp-library)")
    preproc_file = os.path.join(proc_dir, 'preproc.json')
    old_mode, old_ver = None, 0
    if os.path.exists(preproc_file):
        meta = json.load(open(preproc_file, encoding='utf-8'))
        old_mode, old_ver = meta.get('indic_mode'), meta.get('preproc_version', 0)
    mode_changed = os.path.exists(train_csv) and (old_mode != mode or old_ver != PREPROC_VERSION)
    if mode_changed:
        print(f"♻️  Preprocessing changed (mode {old_mode}->{mode}, version {old_ver}->{PREPROC_VERSION}): "
              f"rebuilding data + tokenizer. Old checkpoints of this run no longer match - "
              f"use a new --run-name to keep them.")
    print(f"🔤 Indic mode: {mode}")

    if args.force_reprocess or mode_changed or not (os.path.exists(train_csv) and os.path.exists(dev_csv)):
        files = {}
        if args.asm_bodo_file: files['asm-brx'] = args.asm_bodo_file
        if args.ben_odi_file:  files['ben-odi'] = args.ben_odi_file
        dev_files  = {k: v for k, v in (('asm-brx', args.asm_bodo_dev), ('ben-odi', args.ben_odi_dev)) if v}
        test_files = {k: v for k, v in (('asm-brx', args.asm_bodo_test), ('ben-odi', args.ben_odi_test)) if v}
        info = prepare_all(args.pairs, data_dir=args.data_dir, files=files,
                           dev_files=dev_files, test_files=test_files, out_dir=proc_dir,
                           dev_size=args.dev_size, test_size=args.test_size, seed=args.seed,
                           indic_mode=mode, dedupe_sides=args.dedupe_sides, max_words=args.max_words)
        retrain_spm = True
    else:
        print(f"📂 Re-using processed data in {proc_dir} (use --force-reprocess to rebuild)")
        info = dict(corpus=os.path.join(proc_dir, 'joint_spm_corpus.txt'))
        retrain_spm = False

    if retrain_spm or not os.path.exists(spm_model):
        train_sentencepiece(info['corpus'], spm_prefix, vocab_size=args.vocab_size)

    tokenizer = verify_tokenizer(spm_model)
    report_unk_rate(tokenizer, train_csv)
    vocab_size = tokenizer.vocab_size()

    train_loader = create_dataloader(train_csv, spm_model, batch_size=args.batch_size,
                                     max_len=args.max_len, shuffle=True,
                                     num_workers=args.num_workers, cache_dir=f'cache/{run_name}')
    val_loader = create_dataloader(dev_csv, spm_model, batch_size=args.batch_size,
                                   max_len=args.max_len, shuffle=False,
                                   num_workers=args.num_workers, cache_dir=f'cache/{run_name}')

    n_train = len(train_loader.dataset)
    if n_train < 200_000 and args.dropout < 0.2:
        print(f"💡 Only {n_train:,} training samples: try --dropout 0.3 to reduce over-fitting.")
    return train_loader, val_loader, spm_model, vocab_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    assert args.d_model % args.nhead == 0, "d_model must be divisible by nhead"
    assert args.max_pos >= args.max_len + 2, "--max-pos must be at least --max-len + 2"

    run_name = args.run_name or '+'.join(args.pairs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device} | run: {run_name}")

    train_loader, val_loader, spm_model, vocab_size = setup_data(args, run_name)

    model = TransformerModel(
        vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead, d_ff=args.d_ff,
        num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
        dropout=args.dropout, max_seq_len=args.max_pos,
    )
    initialize_weights(model)
    model = model.to(device)
    test_forward_pass(model, device, vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                                  eps=1e-9, weight_decay=args.weight_decay)
    if args.scheduler == 'warmup':
        w = args.warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / w, math.sqrt(w / (step + 1))))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    config = dict(
        save_dir=f'checkpoints/{run_name}', run_name=run_name, pairs=args.pairs,
        spm_model=spm_model, label_smoothing=args.label_smoothing, patience=args.patience,
        max_grad_norm=args.max_grad_norm, grad_accum_steps=args.grad_accum_steps,
        val_every_n_epochs=args.val_every_n_epochs, amp=args.amp, eval_samples=300,
    )
    trainer = TransformerTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        config=config, tokenizer_path=spm_model,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        trainer.epoch            = ckpt.get('epoch', 0)
        trainer.global_step      = ckpt.get('global_step', 0)
        trainer.best_val_loss    = ckpt.get('best_val_loss', float('inf'))
        trainer.best_bleu        = ckpt.get('best_bleu', 0.0)
        trainer.patience_counter = ckpt.get('patience_counter', 0)
        print(f"🔄 Resumed from epoch {trainer.epoch}")

    print(f"\n🚀 Training up to {args.epochs} epochs\n")
    try:
        trainer.train(num_epochs=args.epochs)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted - saving checkpoint.")
        trainer.save_checkpoint()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✅ Done | best val loss {trainer.best_val_loss:.4f} | best BLEU {trainer.best_bleu:.2f}")
    print(f"   Evaluate with:  python compute_metrics.py --run-name {run_name}")


if __name__ == "__main__":
    main()
