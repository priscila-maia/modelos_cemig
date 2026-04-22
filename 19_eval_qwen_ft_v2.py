#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação do Qwen3-Embedding-0.6B fine-tuned (v2).

Executa dois cenários:
1) Sem cross-encoder (apenas retrieval por embedding)
2) Com cross-encoder (retrieval + rerank)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

SEED = 42
TOP_K_METRICS = [1, 5, 10]
BATCH_SIZE_ENCODE = int(os.getenv("BATCH_SIZE_ENCODE", "64"))
BATCH_SIZE_RERANK = int(os.getenv("BATCH_SIZE_RERANK", "32"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "100"))
ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "1") == "1"

BASE_DIR = Path(__file__).parent
DEFAULT_HF_CACHE_DIR = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", str(DEFAULT_HF_CACHE_DIR)))

EXP_DIR = BASE_DIR / "experiments" / "exp_v2_40k"
EXP_RERANK = BASE_DIR / "experiments" / "exp_v2_40k_reranker_ft"

TEST_FILE = EXP_DIR / "train_pos_v2_test.jsonl"
MODEL_DIR = EXP_DIR / "models" / "qwen3_embedding_0_6b_ft_v2"
RERANKER_MODEL = Path(
    os.getenv(
        "RERANKER_MODEL",
        str(EXP_RERANK / "models" / "reranker_ft"),
    )
)

# Compatibilidade com o fluxo antigo (sem cross).
RESULTS_FILE_LEGACY = EXP_DIR / "results_qwen_ft_v2.json"
METRICS_FILE_LEGACY = EXP_DIR / "metrics_qwen_ft_v2.md"

# Novos artefatos organizados.
RESULTS_FILE_NO_CROSS = EXP_DIR / "results_qwen_ft_v2_no_cross.json"
RESULTS_FILE_CROSS = EXP_DIR / "results_qwen_ft_v2_cross.json"
RESULTS_FILE_COMPARE = EXP_DIR / "results_qwen_ft_v2_compare.json"
METRICS_FILE_COMPARE = EXP_DIR / "metrics_qwen_ft_v2_compare.md"


def setup_cache_dirs():
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR / "sentence_transformers"))
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def load_test_data(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            data.append(json.loads(raw))
    return data


def build_metrics(rankings):
    mrr_scores = calculate_mrr(rankings, k_values=TOP_K_METRICS)
    return {
        **mrr_scores,
        "mean_rank": float(np.mean(rankings)),
        "median_rank": float(np.median(rankings)),
        "top1_accuracy": sum(1 for r in rankings if r == 1) / len(rankings),
        "top5_accuracy": sum(1 for r in rankings if r <= 5) / len(rankings),
        "top10_accuracy": sum(1 for r in rankings if r <= 10) / len(rankings),
    }


def evaluate_no_cross(query_embeddings, context_embeddings, queries):
    rankings = []
    for i in tqdm(range(len(queries)), desc="Retrieval (sem cross)"):
        query_emb = query_embeddings[i]
        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
        sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        rank = np.where(sorted_indices == i)[0][0] + 1
        rankings.append(int(rank))
    return rankings


def evaluate_with_cross(query_embeddings, context_embeddings, queries, contexts, reranker):
    rankings = []
    for i in tqdm(range(len(queries)), desc="Reranking (com cross)"):
        query = queries[i]
        query_emb = query_embeddings[i]
        correct_idx = i

        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
        top_k = min(TOP_K_RETRIEVE, len(contexts))
        top_k_indices = torch.topk(similarities, k=top_k).indices.cpu().numpy()

        pairs = [[query, contexts[idx]] for idx in top_k_indices]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False, batch_size=BATCH_SIZE_RERANK)

        reranked_order = np.argsort(rerank_scores)[::-1]
        reranked_indices = top_k_indices[reranked_order]

        rank = np.where(reranked_indices == correct_idx)[0]
        rank = rank[0] + 1 if len(rank) > 0 else TOP_K_RETRIEVE + 1
        rankings.append(int(rank))
    return rankings


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_mode_markdown(title: str, mode: str, num_pairs: int, total_time: float, metrics):
    return f"""# {title}

**Modo:** {mode}  
**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Conjunto de teste:** {num_pairs:,} pares  
**Tempo total:** {total_time:.1f}s

## Métricas de Ranking

| Métrica | Valor |
|---------|-------|
| MRR@1   | {metrics['MRR@1']:.4f} |
| MRR@5   | {metrics['MRR@5']:.4f} |
| MRR@10  | {metrics['MRR@10']:.4f} |

## Acurácia Top-K

| Top-K | Acurácia |
|-------|----------|
| Top-1 | {metrics['top1_accuracy']:.2%} |
| Top-5 | {metrics['top5_accuracy']:.2%} |
| Top-10 | {metrics['top10_accuracy']:.2%} |
"""


def write_text(path: Path, text: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {TEST_FILE}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Modelo fine-tuned não encontrado: {MODEL_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA não disponível. Esta avaliação foi configurada para usar GPU.")
    if ENABLE_CROSS_ENCODER and not RERANKER_MODEL.exists():
        raise FileNotFoundError(f"Reranker fine-tuned não encontrado: {RERANKER_MODEL}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    setup_cache_dirs()
    sanity_test_mrr()

    device = "cuda"
    model = SentenceTransformer(
        str(MODEL_DIR),
        device=device,
        tokenizer_kwargs={"fix_mistral_regex": True},
    )

    test_data = load_test_data(TEST_FILE)
    queries = [item["query"] for item in test_data]
    contexts = [item["context"] for item in test_data]

    print("Codificando contexts...")
    context_embeddings = model.encode(
        contexts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE_ENCODE,
    )

    print("Codificando queries...")
    query_embeddings = model.encode(
        queries,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE_ENCODE,
    )

    start_no_cross = time.time()
    rankings_no_cross = evaluate_no_cross(query_embeddings, context_embeddings, queries)
    time_no_cross = time.time() - start_no_cross
    metrics_no_cross = build_metrics(rankings_no_cross)

    result_no_cross = {
        "model": "qwen3_embedding_0_6b_ft_v2",
        "model_path": str(MODEL_DIR),
        "mode": "no_cross_encoder",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "num_test_pairs": len(test_data),
        "top_k_metrics": TOP_K_METRICS,
        "total_time_sec": time_no_cross,
        "metrics": metrics_no_cross,
    }

    save_json(RESULTS_FILE_NO_CROSS, result_no_cross)
    save_json(RESULTS_FILE_LEGACY, result_no_cross)
    write_text(
        METRICS_FILE_LEGACY,
        build_mode_markdown(
            "Qwen3-Embedding-0.6B Fine-tuned - Métricas (v2)",
            "Sem Cross-Encoder",
            len(test_data),
            time_no_cross,
            metrics_no_cross,
        ),
    )

    result_cross = None
    if ENABLE_CROSS_ENCODER:
        start_cross = time.time()
        reranker = CrossEncoder(str(RERANKER_MODEL), device=device)
        rankings_cross = evaluate_with_cross(query_embeddings, context_embeddings, queries, contexts, reranker)
        time_cross = time.time() - start_cross
        metrics_cross = build_metrics(rankings_cross)

        result_cross = {
            "model": "qwen3_embedding_0_6b_ft_v2",
            "model_path": str(MODEL_DIR),
            "mode": "with_cross_encoder",
            "reranker_model": str(RERANKER_MODEL),
            "top_k_retrieve": TOP_K_RETRIEVE,
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "num_test_pairs": len(test_data),
            "top_k_metrics": TOP_K_METRICS,
            "total_time_sec": time_cross,
            "metrics": metrics_cross,
        }
        save_json(RESULTS_FILE_CROSS, result_cross)

    compare_payload = {
        "model": "qwen3_embedding_0_6b_ft_v2",
        "model_path": str(MODEL_DIR),
        "seed": SEED,
        "num_test_pairs": len(test_data),
        "no_cross": result_no_cross,
        "with_cross": result_cross,
    }
    save_json(RESULTS_FILE_COMPARE, compare_payload)

    if result_cross is None:
        md_compare = f"""# Qwen FT v2 - Comparação Sem/Com Cross-Encoder

**Conjunto de teste:** {len(test_data):,} pares  
**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Cross-Encoder:** desabilitado (`ENABLE_CROSS_ENCODER=0`)

| Modo | MRR@1 | MRR@5 | MRR@10 | Top-1 | Top-5 | Top-10 |
|------|------:|------:|-------:|------:|------:|-------:|
| Sem Cross-Encoder | {metrics_no_cross['MRR@1']:.4f} | {metrics_no_cross['MRR@5']:.4f} | {metrics_no_cross['MRR@10']:.4f} | {metrics_no_cross['top1_accuracy']:.2%} | {metrics_no_cross['top5_accuracy']:.2%} | {metrics_no_cross['top10_accuracy']:.2%} |
"""
    else:
        metrics_cross = result_cross["metrics"]
        md_compare = f"""# Qwen FT v2 - Comparação Sem/Com Cross-Encoder

**Conjunto de teste:** {len(test_data):,} pares  
**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Reranker (cross):** `{RERANKER_MODEL}`  
**Top-K retrieve para rerank:** {TOP_K_RETRIEVE}

| Modo | MRR@1 | MRR@5 | MRR@10 | Top-1 | Top-5 | Top-10 |
|------|------:|------:|-------:|------:|------:|-------:|
| Sem Cross-Encoder | {metrics_no_cross['MRR@1']:.4f} | {metrics_no_cross['MRR@5']:.4f} | {metrics_no_cross['MRR@10']:.4f} | {metrics_no_cross['top1_accuracy']:.2%} | {metrics_no_cross['top5_accuracy']:.2%} | {metrics_no_cross['top10_accuracy']:.2%} |
| Com Cross-Encoder | {metrics_cross['MRR@1']:.4f} | {metrics_cross['MRR@5']:.4f} | {metrics_cross['MRR@10']:.4f} | {metrics_cross['top1_accuracy']:.2%} | {metrics_cross['top5_accuracy']:.2%} | {metrics_cross['top10_accuracy']:.2%} |
"""

    write_text(METRICS_FILE_COMPARE, md_compare)

    print("=" * 70)
    print("AVALIAÇÃO QWEN FT V2 CONCLUÍDA")
    print("=" * 70)
    print("Sem Cross-Encoder")
    print(f"MRR@1: {metrics_no_cross['MRR@1']:.4f}")
    print(f"MRR@5: {metrics_no_cross['MRR@5']:.4f}")
    print(f"MRR@10: {metrics_no_cross['MRR@10']:.4f}")
    if result_cross is not None:
        metrics_cross = result_cross["metrics"]
        print("-" * 70)
        print("Com Cross-Encoder")
        print(f"MRR@1: {metrics_cross['MRR@1']:.4f}")
        print(f"MRR@5: {metrics_cross['MRR@5']:.4f}")
        print(f"MRR@10: {metrics_cross['MRR@10']:.4f}")
    print("-" * 70)
    print(f"Resultados sem cross: {RESULTS_FILE_NO_CROSS}")
    print(f"Resultados com cross: {RESULTS_FILE_CROSS if result_cross is not None else 'desabilitado'}")
    print(f"Comparação: {RESULTS_FILE_COMPARE}")
    print(f"Métricas comparação: {METRICS_FILE_COMPARE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
