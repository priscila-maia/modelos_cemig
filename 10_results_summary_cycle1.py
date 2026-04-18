#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerar sumário final do ciclo 1 (v1)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXP_DIR = BASE_DIR / "experiments" / "exp_v1_40k"
RESULTS_BASE = EXP_DIR / "results_baseline_e5.json"
RESULTS_FT = EXP_DIR / "results_e5_ft.json"
RESULTS_RERANK = EXP_DIR / "results_reranked.json"
SUMMARY_FILE = EXP_DIR / "results_summary_cycle1_v1.md"


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    m = data["metrics"]
    return {
        "MRR@1": m["MRR@1"],
        "MRR@5": m["MRR@5"],
        "MRR@10": m["MRR@10"],
    }


def main():
    baseline = load_metrics(RESULTS_BASE)
    ft = load_metrics(RESULTS_FT)
    rr = load_metrics(RESULTS_RERANK)

    def delta(a, b):
        return ((b - a) / a) * 100 if a != 0 else 0.0

    md = f"""# Resumo Final - Ciclo 1 (v1)

| Modelo | MRR@1 | MRR@5 | MRR@10 |
|--------|------|------|-------|
| Baseline E5 | {baseline['MRR@1']:.4f} | {baseline['MRR@5']:.4f} | {baseline['MRR@10']:.4f} |
| E5 FT | {ft['MRR@1']:.4f} | {ft['MRR@5']:.4f} | {ft['MRR@10']:.4f} |
| E5 FT + Rerank | {rr['MRR@1']:.4f} | {rr['MRR@5']:.4f} | {rr['MRR@10']:.4f} |

## Deltas (%)

- Baseline → FT: MRR@1 {delta(baseline['MRR@1'], ft['MRR@1']):.2f}%, MRR@5 {delta(baseline['MRR@5'], ft['MRR@5']):.2f}%, MRR@10 {delta(baseline['MRR@10'], ft['MRR@10']):.2f}%
- FT → Rerank: MRR@1 {delta(ft['MRR@1'], rr['MRR@1']):.2f}%, MRR@5 {delta(ft['MRR@5'], rr['MRR@5']):.2f}%, MRR@10 {delta(ft['MRR@10'], rr['MRR@10']):.2f}%
"""

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        f.write(md)

    print("=" * 70)
    print("RESUMO FINAL (V1) GERADO")
    print("=" * 70)
    print(md)
    print("=" * 70)


if __name__ == "__main__":
    main()
