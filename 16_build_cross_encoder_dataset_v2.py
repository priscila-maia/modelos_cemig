#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerar dataset para treino de Cross-Encoder (reranker)
1 positivo + 1 negativo por query.
"""

import json
import random
import time
from pathlib import Path

from tqdm import tqdm

SEED = 42

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
INPUT_FILE = DATA_DIR / "train_pos_v2.jsonl"
OUTPUT_FILE = DATA_DIR / "train_cross_encoder.jsonl"
REPORT_FILE = BASE_DIR / "build_cross_encoder_report.json"


def build_offsets(path: Path):
    offsets = []
    total_lines = 0
    with path.open("r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            total_lines += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ctx = obj.get("context")
            if isinstance(ctx, str) and ctx.strip():
                offsets.append(pos)
    return offsets, total_lines


def read_context_at(f, offset):
    f.seek(offset)
    line = f.readline()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    ctx = obj.get("context")
    if isinstance(ctx, str) and ctx.strip():
        return ctx
    return None


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    random.seed(SEED)
    start = time.time()

    offsets, total_lines = build_offsets(INPUT_FILE)
    if not offsets:
        raise RuntimeError("Nenhum contexto válido encontrado para amostragem.")

    positives_written = 0
    negatives_written = 0

    with INPUT_FILE.open("r", encoding="utf-8") as fin, INPUT_FILE.open("r", encoding="utf-8") as frand, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for _ in tqdm(range(total_lines), desc="Gerando", unit="linha"):
            current_offset = fin.tell()
            line = fin.readline()
            if not line:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            query = obj.get("query")
            context = obj.get("context")
            if not isinstance(query, str) or not query.strip():
                continue
            if not isinstance(context, str) or not context.strip():
                continue

            # Positivo
            pos = {
                "query": query,
                "context": context,
                "label": 1,
            }
            fout.write(json.dumps(pos, ensure_ascii=False) + "\n")
            positives_written += 1

            # Negativo
            neg_ctx = None
            for _ in range(10):
                neg_offset = random.choice(offsets)
                if neg_offset == current_offset:
                    continue
                neg_ctx = read_context_at(frand, neg_offset)
                if neg_ctx:
                    break

            if neg_ctx:
                neg = {
                    "query": query,
                    "context": neg_ctx,
                    "label": 0,
                }
                fout.write(json.dumps(neg, ensure_ascii=False) + "\n")
                negatives_written += 1

    report = {
        "total_positivos": positives_written,
        "total_negativos": negatives_written,
        "seed": SEED,
        "tempo_total": time.time() - start,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("BUILD CROSS-ENCODER DATASET")
    print("=" * 70)
    print(f"Positivos: {positives_written}")
    print(f"Negativos: {negatives_written}")
    print(f"Saída: {OUTPUT_FILE}")
    print(f"Relatório: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
