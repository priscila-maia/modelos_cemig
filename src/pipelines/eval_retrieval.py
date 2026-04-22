"""Generic retrieval evaluation pipeline with optional cross-encoder rerank."""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.core.cache import setup_hf_cache_dirs
from src.core.io import write_json, write_text
from src.core.metrics import build_rank_metrics, sanity_test_mrr
from src.core.seed import set_all_seeds
from src.data.jsonl import load_query_context_pairs
from src.pipelines.profiles import get_profile_module
from src.retrieval.encoder import (
    encode_texts,
    load_sentence_encoder,
    rank_indices_by_cosine,
    topk_indices_by_cosine,
)
from src.retrieval.rerank import load_cross_encoder, rerank_indices


def _markdown_table(metrics_no_cross, metrics_cross=None):
    if metrics_cross is None:
        return f"""| Modo | MRR@1 | MRR@5 | MRR@10 | Top-1 | Top-5 | Top-10 |
|------|------:|------:|-------:|------:|------:|-------:|
| Sem Cross-Encoder | {metrics_no_cross['MRR@1']:.4f} | {metrics_no_cross['MRR@5']:.4f} | {metrics_no_cross['MRR@10']:.4f} | {metrics_no_cross['top1_accuracy']:.2%} | {metrics_no_cross['top5_accuracy']:.2%} | {metrics_no_cross['top10_accuracy']:.2%} |
"""

    return f"""| Modo | MRR@1 | MRR@5 | MRR@10 | Top-1 | Top-5 | Top-10 |
|------|------:|------:|-------:|------:|------:|-------:|
| Sem Cross-Encoder | {metrics_no_cross['MRR@1']:.4f} | {metrics_no_cross['MRR@5']:.4f} | {metrics_no_cross['MRR@10']:.4f} | {metrics_no_cross['top1_accuracy']:.2%} | {metrics_no_cross['top5_accuracy']:.2%} | {metrics_no_cross['top10_accuracy']:.2%} |
| Com Cross-Encoder | {metrics_cross['MRR@1']:.4f} | {metrics_cross['MRR@5']:.4f} | {metrics_cross['MRR@10']:.4f} | {metrics_cross['top1_accuracy']:.2%} | {metrics_cross['top5_accuracy']:.2%} | {metrics_cross['top10_accuracy']:.2%} |
"""


def _eval_no_cross(query_embeddings, context_embeddings):
    rankings = []
    for i in tqdm(range(query_embeddings.shape[0]), desc="Retrieval (no cross)"):
        sorted_indices = rank_indices_by_cosine(query_embeddings[i], context_embeddings)
        rank = int(np.where(sorted_indices == i)[0][0] + 1)
        rankings.append(rank)
    return rankings


def _eval_with_cross(query_embeddings, context_embeddings, queries, contexts, reranker, top_k_retrieve, batch_size_rerank):
    rankings = []
    for i in tqdm(range(query_embeddings.shape[0]), desc="Reranking (with cross)"):
        top_k = min(top_k_retrieve, len(contexts))
        candidate_indices = topk_indices_by_cosine(query_embeddings[i], context_embeddings, k=top_k)
        reranked_indices = rerank_indices(
            query=queries[i],
            contexts=contexts,
            candidate_indices=candidate_indices,
            reranker=reranker,
            batch_size=batch_size_rerank,
        )
        rank_arr = np.where(reranked_indices == i)[0]
        rank = int(rank_arr[0] + 1) if len(rank_arr) > 0 else top_k_retrieve + 1
        rankings.append(rank)
    return rankings


def run_eval_retrieval(profile_name: str):
    profile = get_profile_module(profile_name)
    cfg = profile.eval_retrieval_config()

    test_file = cfg["test_file"]
    encoder_model = cfg["encoder_model"]
    reranker_model = str(cfg["reranker_model"])

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    if not encoder_model.exists():
        raise FileNotFoundError(f"Encoder model not found: {encoder_model}")
    if cfg.get("require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation profile.")
    if cfg["enable_cross_encoder"] and not torch.cuda.is_available():
        raise RuntimeError("Cross-encoder mode requires CUDA in this profile.")

    reranker_is_repo_id = "/" in reranker_model and not reranker_model.startswith(("/", "."))
    reranker_path = Path(reranker_model)
    if cfg["enable_cross_encoder"] and not reranker_is_repo_id and not reranker_path.exists():
        raise FileNotFoundError(f"Reranker model not found: {reranker_model}")

    setup_hf_cache_dirs()
    set_all_seeds(cfg["seed"])
    sanity_test_mrr(cfg["top_k_metrics"])

    pairs = load_query_context_pairs(test_file)
    queries = [p["query"] for p in pairs]
    contexts = [p["context"] for p in pairs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_sentence_encoder(
        str(encoder_model),
        device=device,
        trust_remote_code=False,
        tokenizer_kwargs=cfg.get("tokenizer_kwargs"),
    )

    print("Encoding contexts...")
    context_embeddings = encode_texts(encoder, contexts, batch_size=cfg["batch_size_encode"])
    print("Encoding queries...")
    query_embeddings = encode_texts(encoder, queries, batch_size=cfg["batch_size_encode"])

    start_no_cross = time.time()
    rankings_no_cross = _eval_no_cross(query_embeddings, context_embeddings)
    no_cross_time = time.time() - start_no_cross
    metrics_no_cross = build_rank_metrics(rankings_no_cross, cfg["top_k_metrics"])

    no_cross_payload = {
        "profile": profile_name,
        "mode": "no_cross_encoder",
        "model": encoder_model.name,
        "model_path": str(encoder_model),
        "timestamp": datetime.now().isoformat(),
        "seed": cfg["seed"],
        "num_test_pairs": len(pairs),
        "top_k_metrics": cfg["top_k_metrics"],
        "total_time_sec": no_cross_time,
        "metrics": metrics_no_cross,
    }
    write_json(cfg["results_no_cross"], no_cross_payload)
    write_json(cfg["results_legacy"], no_cross_payload)

    write_text(
        cfg["metrics_legacy"],
        f"""# Retrieval Metrics

**Profile:** {profile_name}  
**Mode:** Sem Cross-Encoder  
**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Test pairs:** {len(pairs):,}  
**Time:** {no_cross_time:.1f}s

{_markdown_table(metrics_no_cross)}
""",
    )

    cross_payload = None
    metrics_cross = None
    if cfg["enable_cross_encoder"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Cross mode requires CUDA in this profile.")

        start_cross = time.time()
        reranker = load_cross_encoder(reranker_model, device=device)
        rankings_cross = _eval_with_cross(
            query_embeddings,
            context_embeddings,
            queries,
            contexts,
            reranker,
            cfg["top_k_retrieve"],
            cfg["batch_size_rerank"],
        )
        cross_time = time.time() - start_cross
        metrics_cross = build_rank_metrics(rankings_cross, cfg["top_k_metrics"])

        cross_payload = {
            "profile": profile_name,
            "mode": "with_cross_encoder",
            "model": encoder_model.name,
            "model_path": str(encoder_model),
            "reranker_model": reranker_model,
            "top_k_retrieve": cfg["top_k_retrieve"],
            "timestamp": datetime.now().isoformat(),
            "seed": cfg["seed"],
            "num_test_pairs": len(pairs),
            "top_k_metrics": cfg["top_k_metrics"],
            "total_time_sec": cross_time,
            "metrics": metrics_cross,
        }
        write_json(cfg["results_cross"], cross_payload)

    compare_payload = {
        "profile": profile_name,
        "model_path": str(encoder_model),
        "seed": cfg["seed"],
        "num_test_pairs": len(pairs),
        "no_cross": no_cross_payload,
        "with_cross": cross_payload,
    }
    write_json(cfg["results_compare"], compare_payload)

    write_text(
        cfg["metrics_compare"],
        f"""# Retrieval Comparison

**Profile:** {profile_name}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Test pairs:** {len(pairs):,}

{_markdown_table(metrics_no_cross, metrics_cross)}
""",
    )

    print("=" * 70)
    print("EVAL RETRIEVAL COMPLETED")
    print("=" * 70)
    print(f"Profile: {profile_name}")
    print(f"No cross MRR@1: {metrics_no_cross['MRR@1']:.4f}")
    if metrics_cross is not None:
        print(f"With cross MRR@1: {metrics_cross['MRR@1']:.4f}")
    print(f"Compare file: {cfg['results_compare']}")
    print(f"Metrics file: {cfg['metrics_compare']}")
    print("=" * 70)
