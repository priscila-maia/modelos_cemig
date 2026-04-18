#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação E5-Large fine-tuned (v2)
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

SEED = 42
TOP_K_METRICS = [1, 5, 10]
BATCH_SIZE = 64

BASE_DIR = Path(__file__).parent
EXP_DIR = BASE_DIR / "experiments" / "exp_v2_40k"
TEST_FILE = EXP_DIR / "train_pos_v2_test.jsonl"
MODEL_DIR = EXP_DIR / "models" / "e5_large_ft_v2"
RESULTS_FILE = EXP_DIR / "results_e5_ft_v2.json"
METRICS_FILE = EXP_DIR / "metrics_e5_ft_v2.md"


def calculate_mrr(rankings, k_values=[1, 5, 10]):
    mrr_scores = {}
    for k in k_values:
        reciprocal_ranks = []
        for rank in rankings:
            reciprocal_ranks.append(1.0 / rank if rank <= k else 0.0)
        mrr_scores[f"MRR@{k}"] = np.mean(reciprocal_ranks)
    return mrr_scores


def sanity_test_mrr():
    mrr_1 = calculate_mrr([1], k_values=TOP_K_METRICS)
    if abs(mrr_1["MRR@1"] - 1.0) > 1e-9:
        raise ValueError("Sanity test falhou: rank=1")

    mrr_2 = calculate_mrr([2], k_values=TOP_K_METRICS)
    if not (abs(mrr_2["MRR@1"] - 0.0) < 1e-9 and abs(mrr_2["MRR@5"] - 0.5) < 1e-9):
        raise ValueError("Sanity test falhou: rank=2")

    mrr_3 = calculate_mrr([999], k_values=TOP_K_METRICS)
    if not (mrr_3["MRR@1"] == 0.0 and mrr_3["MRR@5"] == 0.0 and mrr_3["MRR@10"] == 0.0):
        raise ValueError("Sanity test falhou: rank grande")


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {TEST_FILE}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Modelo fine-tuned não encontrado: {MODEL_DIR}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sanity_test_mrr()
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(MODEL_DIR), device=device)

    test_data = []
    with TEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            test_data.append(json.loads(raw))

    queries = [item["query"] for item in test_data]
    contexts = [item["context"] for item in test_data]

    context_embeddings = model.encode(
        contexts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )

    query_embeddings = model.encode(
        queries,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )

    rankings = []
    for i in tqdm(range(len(queries)), desc="Retrieval"):
        query_emb = query_embeddings[i]
        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
        sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        rank = np.where(sorted_indices == i)[0][0] + 1
        rankings.append(rank)

    mrr_scores = calculate_mrr(rankings, k_values=TOP_K_METRICS)
    mean_rank = float(np.mean(rankings))
    median_rank = float(np.median(rankings))
    top1_acc = sum(1 for r in rankings if r == 1) / len(rankings)
    top5_acc = sum(1 for r in rankings if r <= 5) / len(rankings)
    top10_acc = sum(1 for r in rankings if r <= 10) / len(rankings)

    total_time = time.time() - start

    results = {
        "model": "e5_large_ft_v2",
        "model_path": str(MODEL_DIR),
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "num_test_pairs": len(test_data),
        "top_k_metrics": TOP_K_METRICS,
        "total_time_sec": total_time,
        "metrics": {
            **mrr_scores,
            "mean_rank": mean_rank,
            "median_rank": median_rank,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "top10_accuracy": top10_acc,
        },
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md = f"""# E5-Large Fine-tuned - Métricas (v2)

**Modelo:** E5-Large Fine-tuned  
**Caminho:** `{MODEL_DIR.name}`  
**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Conjunto de teste:** {len(test_data):,} pares  
**Tempo total:** {total_time:.1f}s

## Métricas de Ranking

| Métrica | Valor |
|---------|-------|
| MRR@1   | {mrr_scores['MRR@1']:.4f} |
| MRR@5   | {mrr_scores['MRR@5']:.4f} |
| MRR@10  | {mrr_scores['MRR@10']:.4f} |

## Acurácia Top-K

| Top-K | Acurácia |
|-------|----------|
| Top-1 | {top1_acc:.2%} |
| Top-5 | {top5_acc:.2%} |
| Top-10 | {top10_acc:.2%} |
"""

    with METRICS_FILE.open("w", encoding="utf-8") as f:
        f.write(md)

    print("=" * 70)
    print("AVALIAÇÃO E5-FT V2 CONCLUÍDA")
    print("=" * 70)
    print(f"MRR@1: {mrr_scores['MRR@1']:.4f}")
    print(f"MRR@5: {mrr_scores['MRR@5']:.4f}")
    print(f"MRR@10: {mrr_scores['MRR@10']:.4f}")
    print(f"Tempo total: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
