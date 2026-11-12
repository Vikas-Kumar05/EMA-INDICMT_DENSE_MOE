"""
compare_runs.py - Side-by-side table of several evaluated runs


"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--results-dir', default='evaluation_results')
    ap.add_argument('--metrics', nargs='+', default=['BLEU_13a', 'spBLEU', 'chrF++', 'TER', 'CER'])
    args = ap.parse_args()

    data = {}
    for run in args.runs:
        path = Path(args.results_dir) / run / 'results.json'
        if not path.exists():
            raise SystemExit(f"❌ {path} not found - run: python compute_metrics.py --run-name {run}")
        data[run] = json.loads(path.read_text(encoding='utf-8'))

    directions = list(next(iter(data.values()))['directions'])
    width = max(len(r) for r in args.runs) + 2

    for tag in directions:
        label = data[args.runs[0]]['directions'][tag]['label']
        print(f"\n{label}")
        print(f"{'run':<{width}}" + "".join(f"{m:>10}" for m in args.metrics) + f"{'params':>14}")
        for run in args.runs:
            d = data[run]['directions'].get(tag)
            if d is None:
                continue
            vals = "".join(f"{str(d['metrics'].get(m, 'N/A')):>10}" for m in args.metrics)
            print(f"{run:<{width}}{vals}{data[run].get('parameters', 0):>14,}")

    print("\n(BLEU / chrF++ / spBLEU: higher is better; TER / CER: lower is better)")


if __name__ == '__main__':
    main()
