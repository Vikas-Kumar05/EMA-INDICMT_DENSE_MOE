"""
translate.py 

"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm

from text_utils import LANGS, PAIRS, normalize_text
from indic_utils import IndicTransPreprocessor
from inference import DecodeConfig, load_model, Translator


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--src', required=True, choices=list(LANGS))
    ap.add_argument('--tgt', required=True, choices=list(LANGS))
    ap.add_argument('--text', default=None, help='one sentence to translate')
    ap.add_argument('--input', default=None, help='text file, one sentence per line (default: stdin)')
    ap.add_argument('--output', default=None, help='output file (default: stdout)')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--beam-size', type=int, default=DecodeConfig.beam_size)
    ap.add_argument('--digits', choices=['native', 'ascii', 'keep'], default='native')
    ap.add_argument('--latin-policy', choices=['source', 'keep', 'drop'], default='source')
    ap.add_argument('--no-nukta-recompose', action='store_true')
    args = ap.parse_args()

    if args.src == args.tgt:
        sys.exit("❌ --src and --tgt must differ")
    if not any({args.src, args.tgt} == set(info['langs']) for info in PAIRS.values()):
        sys.exit("❌ direction not supported: use asm<->brx or ben<->odi")

    cfg = DecodeConfig()
    cfg.beam_size, cfg.digits, cfg.latin_policy = args.beam_size, args.digits, args.latin_policy
    cfg.recompose_nukta = not args.no_nukta_recompose

    model, train_cfg = load_model(args.checkpoint or f"checkpoints/{args.run_name}/best_model.pt", cfg.device)
    data_name = train_cfg.get('data_name', args.run_name)
    spm_path = args.tokenizer or train_cfg.get('spm_model') or f"tokenizer/{data_name}/spm.model"
    tokenizer = spm.SentencePieceProcessor(model_file=spm_path)

    cfg.use_indictrans = bool(train_cfg.get('use_indictrans', False))
    cfg.indic_placeholders = bool(train_cfg.get('indic_placeholders', False))
    indic = IndicTransPreprocessor(placeholders=cfg.indic_placeholders) if cfg.use_indictrans else None
    translator = Translator(model, tokenizer, cfg, indic)

    if args.text is not None:
        lines = [args.text]
    elif args.input:
        lines = Path(args.input).read_text(encoding='utf-8').splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    out = []
    for line in tqdm(lines, desc='Translating', disable=len(lines) < 5, file=sys.stderr):
        line = normalize_text(line)
        out.append(translator.translate(line, args.src, args.tgt) if line else '')

    if args.output:
        Path(args.output).write_text('\n'.join(out) + '\n', encoding='utf-8')
        print(f"✅ {len(out)} sentences -> {args.output}", file=sys.stderr)
    else:
        print('\n'.join(out))


if __name__ == '__main__':
    main()
