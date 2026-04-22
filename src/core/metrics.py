"""Metrics shared by retrieval and downstream evaluation."""

from typing import Dict, Iterable, List

import numpy as np


def calculate_mrr(rankings: Iterable[int], k_values: List[int]) -> Dict[str, float]:
    rankings = list(rankings)
    mrr_scores = {}
    for k in k_values:
        reciprocal_ranks = [1.0 / rank if rank <= k else 0.0 for rank in rankings]
        mrr_scores[f"MRR@{k}"] = float(np.mean(reciprocal_ranks))
    return mrr_scores


def sanity_test_mrr(k_values: List[int]) -> None:
    mrr_1 = calculate_mrr([1], k_values=k_values)
    if abs(mrr_1[f"MRR@{k_values[0]}"] - 1.0) > 1e-9:
        raise ValueError("MRR sanity test failed for rank=1")

    mrr_2 = calculate_mrr([2], k_values=k_values)
    if mrr_2.get("MRR@1", 0.0) != 0.0:
        raise ValueError("MRR sanity test failed for rank=2 at @1")


def build_rank_metrics(rankings: Iterable[int], k_values: List[int]) -> Dict[str, float]:
    rankings = list(rankings)
    out = calculate_mrr(rankings, k_values=k_values)
    out["mean_rank"] = float(np.mean(rankings))
    out["median_rank"] = float(np.median(rankings))
    out["top1_accuracy"] = sum(1 for r in rankings if r == 1) / len(rankings)
    out["top5_accuracy"] = sum(1 for r in rankings if r <= 5) / len(rankings)
    out["top10_accuracy"] = sum(1 for r in rankings if r <= 10) / len(rankings)
    return out
