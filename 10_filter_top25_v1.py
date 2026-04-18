#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtro Top-25 no train_pos_v1.jsonl (Ciclo 2 - base v1)
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

SEED = 42
TOP_K = 25
BATCH_SIZE = 64

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
INPUT_FILE = DATA_DIR / "train_pos_v1.jsonl"
OUTPUT_FILE = DATA_DIR / "train_pos_v2.jsonl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "filter_top25_v1_report.json"
MODEL_DIR = BASE_DIR / "experiments" / "exp_v1_40k" / "models" / "e5_large_ft"


def load_contexts(input_path: Path) -> List[str]:
    contexts = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            context = obj.get("context")
            if isinstance(context, str) and context:
                contexts.append(context)
    return contexts


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Modelo fine-tuned não encontrado: {MODEL_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carregar encoder
    encoder = SentenceTransformer(str(MODEL_DIR), device=device)

    # Passo 1: carregar contexts e gerar embeddings (uma vez)
    contexts = load_contexts(INPUT_FILE)
    if not contexts:
        raise RuntimeError("Nenhum contexto válido encontrado no dataset de entrada.")

    context_embeddings = encoder.encode(
        contexts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )

    total_exemplos = 0
    mantidos = 0
    removidos = 0

    # Passo 2: streaming e filtro
    batch_queries: List[str] = []
    batch_rows: List[Dict[str, Any]] = []

    with INPUT_FILE.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Filtrando", unit="linha"):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            query = obj.get("query")
            context_gold = obj.get("context")

            if not isinstance(query, str) or not query.strip():
                continue
            if not isinstance(context_gold, str) or not context_gold.strip():
                continue

            total_exemplos += 1
            batch_queries.append(query)
            batch_rows.append(obj)

            if len(batch_queries) >= BATCH_SIZE:
                _process_batch(batch_queries, batch_rows, context_embeddings, contexts, fout)
                mantidos += sum(1 for r in batch_rows if r.get("_kept") is True)
                removidos += sum(1 for r in batch_rows if r.get("_kept") is False)
                batch_queries.clear()
                batch_rows.clear()

        if batch_queries:
            _process_batch(batch_queries, batch_rows, context_embeddings, contexts, fout)
            mantidos += sum(1 for r in batch_rows if r.get("_kept") is True)
            removidos += sum(1 for r in batch_rows if r.get("_kept") is False)

    tempo_total = time.time() - start
    percentual_removido = (removidos / total_exemplos * 100) if total_exemplos > 0 else 0

    report = {
        "total_exemplos": total_exemplos,
        "mantidos": mantidos,
        "removidos": removidos,
        "percentual_removido": percentual_removido,
        "top_k_usado": TOP_K,
        "seed": SEED,
        "tempo_total": tempo_total,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("FILTRO TOP-25 (NOVO) CONCLUÍDO")
    print("=" * 70)
    print(f"Total: {total_exemplos}")
    print(f"Mantidos: {mantidos}")
    print(f"Removidos: {removidos} ({percentual_removido:.1f}%)")
    print(f"Tempo total: {tempo_total:.1f}s")
    print(f"Saída: {OUTPUT_FILE}")
    print(f"Relatório: {REPORT_FILE}")
    print("=" * 70)
    print(f"Filtro Top-25 concluído: {mantidos} mantidos, {removidos} removidos ({percentual_removido:.1f}%).")


def _process_batch(batch_queries, batch_rows, context_embeddings, contexts, fout):
    query_embeddings = encoder_encode(batch_queries, context_embeddings.device)
    for i, query_emb in enumerate(query_embeddings):
        row = batch_rows[i]
        context_gold = row.get("context")

        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
        top_k_indices = torch.topk(similarities, k=min(TOP_K, len(contexts))).indices.cpu().numpy()

        # Verifica se o context_gold está no top-25 (por igualdade de string)
        kept = any(contexts[idx] == context_gold for idx in top_k_indices)
        row["_kept"] = kept
        if kept:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def encoder_encode(queries, device):
    # Evita re-instanciar o encoder
    global _ENCODER
    if "_ENCODER" not in globals():
        _ENCODER = SentenceTransformer(str(MODEL_DIR), device=device)
    return _ENCODER.encode(
        queries,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
