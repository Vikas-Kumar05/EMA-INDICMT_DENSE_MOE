"""
main.py - Train a Transformer for  Assamese<->Bodo  and/or  Bengali<->Odia


"""

import os
import math
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from text_utils import PAIRS
from indic_utils import indictrans_available, INSTALL_HINT
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
    p.add_argument('--run-name', default=None,
                   help='checkpoint / evaluation folder. default: pairs joined with "+" '
                        '(+ "_indictrans" / "_moe" suffixes)')
    p.add_argument('--dev-size', type=int, default=1000)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--vocab-size', type=int, default=16000)
    p.add_argument('--max-len', type=int, default=128, help='max tokens per training sentence')
    p.add_argument('--max-pos', type=int, default=256, help='positional table size (>= max-len+2)')
    p.add_argument('--min-script-ratio', type=float, default=0.5,
                   help='keep a sentence only if >= this share of its letters is in the language\'s own '
                        'script. 0.5 = old behaviour; 0.7-0.8 removes rows full of English, so the model '
                        'learns to write less English (needs --force-reprocess)')
    p.add_argument('--no-byte-fallback', action='store_true',
                   help='train SentencePiece WITHOUT byte fallback (rare symbols become <unk>/"⁇")')
    p.add_argument('--force-reprocess', action='store_true')
    p.add_argument('--num-workers', type=int, default=2)

    # IndicTrans toolkit pre/post-processing
    p.add_argument('--indictrans', action='store_true',
                   help='normalise/tokenise/transliterate text with the IndicTrans toolkit '
                        '(pre-processing for training, post-processing at evaluation)')
    p.add_argument('--indic-placeholders', action='store_true',
                   help='replace URLs/e-mails/long numerals by <IDn> placeholders (off by default)')

    # model
    p.add_argument('--d-model', type=int, default=512)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--enc-layers', type=int, default=6)
    p.add_argument('--dec-layers', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.1)

    # Mixture-of-Experts
    p.add_argument('--moe', action='store_true',
                   help='replace the dense FFN of every --moe-every-th layer (encoder AND decoder) '
                        'by a sparse top-k Mixture-of-Experts')
    p.add_argument('--moe-experts', type=int, default=4, help='experts per MoE layer')
    p.add_argument('--moe-top-k', type=int, default=2, help='experts used per token')
    p.add_argument('--moe-every', type=int, default=2, help='MoE on every N-th layer (2 = 2nd,4th,6th)')
    p.add_argument('--moe-aux-weight', type=float, default=0.01, help='load-balancing loss weight')
    p.add_argument('--moe-z-weight', type=float, default=0.001, help='router z-loss weight')
    p.add_argument('--moe-jitter', type=float, default=0.0, help='multiplicative router-input noise (train only)')

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
    SwiGLU feed-forward layers (dense FFN and every MoE expert) a Xavier init; MoE routers
    start near-uniform (small normal) so all experts get tokens at the beginning.

    The previous version re-initialised EVERY nn.Linear - including the output
    projection, whose weight is TIED to the embedding matrix. That overwrote the
    embeddings with a much smaller scale and un-zeroed the padding row.
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and '.ffn.' in name:
            if name.endswith('.router'):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
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

def setup_data(args, data_name):
    """Data / tokenizer / cache folders are keyed by `data_name`, so a baseline and a
    MoE run on the same text share one processed set (and one SentencePiece model)."""
    run_name = data_name
    proc_dir = os.path.join('processed', run_name)
    # tokenizers with / without byte fallback live in different folders (never mixed up)
    spm_prefix = os.path.join('tokenizer', run_name + ('' if args.no_byte_fallback else '.bf'), 'spm')
    spm_model = spm_prefix + '.model'
    train_csv = os.path.join(proc_dir, 'bidir_train.csv')
    dev_csv = os.path.join(proc_dir, 'bidir_dev.csv')

    if args.force_reprocess or not (os.path.exists(train_csv) and os.path.exists(dev_csv)):
        files = {}
        if args.asm_bodo_file: files['asm-brx'] = args.asm_bodo_file
        if args.ben_odi_file:  files['ben-odi'] = args.ben_odi_file
        dev_files  = {k: v for k, v in (('asm-brx', args.asm_bodo_dev), ('ben-odi', args.ben_odi_dev)) if v}
        test_files = {k: v for k, v in (('asm-brx', args.asm_bodo_test), ('ben-odi', args.ben_odi_test)) if v}
        info = prepare_all(args.pairs, data_dir=args.data_dir, files=files,
                           dev_files=dev_files, test_files=test_files, out_dir=proc_dir,
                           dev_size=args.dev_size, test_size=args.test_size, seed=args.seed,
                           min_script_ratio=args.min_script_ratio,
                           indictrans=args.indictrans, indic_placeholders=args.indic_placeholders)
        retrain_spm = True
    else:
        print(f"📂 Re-using processed data in {proc_dir} (use --force-reprocess to rebuild)")
        info = dict(corpus=os.path.join(proc_dir, 'joint_spm_corpus.txt'))
        retrain_spm = False

    if retrain_spm or not os.path.exists(spm_model):
        train_sentencepiece(info['corpus'], spm_prefix, vocab_size=args.vocab_size,
                            byte_fallback=not args.no_byte_fallback)

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

    if args.moe and not 1 <= args.moe_top_k <= args.moe_experts:
        raise SystemExit("--moe-top-k must be between 1 and --moe-experts")
    if args.indictrans and not indictrans_available():
        raise SystemExit(f"❌ {INSTALL_HINT}")

    # data_name: identifies the preprocessed data + tokenizer (depends on text processing only)
    data_name = '+'.join(args.pairs) + ('.indictrans' if args.indictrans else '')
    data_name += '.ph' if (args.indictrans and args.indic_placeholders) else ''
    # run_name: identifies the checkpoints / logs / evaluation (depends on everything)
    run_name = args.run_name or ('+'.join(args.pairs)
                                 + ('_indictrans' if args.indictrans else '')
                                 + ('_moe' if args.moe else ''))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device} | run: {run_name} | data: {data_name}")
    print(f"   IndicTrans toolkit : {'ON' if args.indictrans else 'off'}"
          f" | MoE : {('top-%d of %d experts, every %d layers' % (args.moe_top_k, args.moe_experts, args.moe_every)) if args.moe else 'off (dense baseline)'}")

    train_loader, val_loader, spm_model, vocab_size = setup_data(args, data_name)

    model = TransformerModel(
        vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead, d_ff=args.d_ff,
        num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
        dropout=args.dropout, max_seq_len=args.max_pos,
        moe_every=args.moe_every if args.moe else 0,
        moe_num_experts=args.moe_experts, moe_top_k=args.moe_top_k, moe_jitter=args.moe_jitter,
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
        # needed by compute_metrics.py to rebuild the exact evaluation pipeline
        data_name=data_name, use_indictrans=args.indictrans,
        indic_placeholders=args.indic_placeholders, use_moe=args.moe,
        moe_aux_weight=args.moe_aux_weight, moe_z_weight=args.moe_z_weight,
    )
    trainer = TransformerTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        config=config, tokenizer_path=spm_model,
    )

    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:                                   # older torch
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
