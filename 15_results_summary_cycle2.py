#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumo final do Ciclo 2 (v2)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXP_V2 = BASE_DIR / "experiments" / "exp_v2_40k"
EXP_V1 = BASE_DIR / "experiments" / "exp_v1_40k"

RESULTS_V2_FT = EXP_V2 / "results_e5_ft_v2.json"
RESULTS_V2_RR = EXP_V2 / "results_reranked_v2.json"
SUMMARY_FILE = EXP_V2 / "results_summary_cycle2.md"

RESULTS_V1_FT = EXP_V1 / "results_e5_ft.json"
RESULTS_V1_RR = EXP_V1 / "results_reranked.json"


def load_metrics(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    m = data["metrics"]
    return {
        "MRR@1": m.get("MRR@1"),
        "MRR@5": m.get("MRR@5"),
        "MRR@10": m.get("MRR@10"),
    }


def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def main():
    v2_ft = load_metrics(RESULTS_V2_FT)
    v2_rr = load_metrics(RESULTS_V2_RR)
    v1_ft = load_metrics(RESULTS_V1_FT)
    v1_rr = load_metrics(RESULTS_V1_RR)

    md = f"""# Resumo Final - Ciclo 2 (v2)

## Métricas v2

| Modelo | MRR@1 | MRR@5 | MRR@10 |
|--------|------|------|-------|
| E5 FT v2 | {fmt(v2_ft['MRR@1'] if v2_ft else None)} | {fmt(v2_ft['MRR@5'] if v2_ft else None)} | {fmt(v2_ft['MRR@10'] if v2_ft else None)} |
| E5 FT v2 + Rerank | {fmt(v2_rr['MRR@1'] if v2_rr else None)} | {fmt(v2_rr['MRR@5'] if v2_rr else None)} | {fmt(v2_rr['MRR@10'] if v2_rr else None)} |

## Comparação com Ciclo 1 (exp_v1_40k)

| Modelo | MRR@1 | MRR@5 | MRR@10 |
|--------|------|------|-------|
| E5 FT (ciclo 1) | {fmt(v1_ft['MRR@1'] if v1_ft else None)} | {fmt(v1_ft['MRR@5'] if v1_ft else None)} | {fmt(v1_ft['MRR@10'] if v1_ft else None)} |
| E5 FT + Rerank (ciclo 1) | {fmt(v1_rr['MRR@1'] if v1_rr else None)} | {fmt(v1_rr['MRR@5'] if v1_rr else None)} | {fmt(v1_rr['MRR@10'] if v1_rr else None)} |

## Observações

- v2 mais limpo; espera-se métrica mais confiável.
- Seed: 42
- Artefatos v2 em: {EXP_V2}
"""

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        f.write(md)

    print("=" * 70)
    print("RESUMO CICLO 2 GERADO")
    print("=" * 70)
    print(md)
    print("=" * 70)


if __name__ == "__main__":
    main()
