#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enriquecimento do snapshot snapshot_v1.jsonl com chunk_text e chunk_len.
Reconstrói chunks com a mesma lógica do gerador original.
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.append("/raid/user_priscilaribeiro/Cemig")

from src.dataset import get_dataset  # noqa: E402


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
INPUT_FILE = DATA_DIR / "snapshot_v1.jsonl"
OUTPUT_FILE = DATA_DIR / "snapshot_v1_enriquecido.jsonl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "enrich_v1_report.json"

DATASET_NAME = "cemig-ceia/energy_dataset_v1"
CHUNK_SIZE = 1024
MIN_CHUNK_LEN = 100


def split_into_chunks(text: str, chunk_size: int = 1024):
    """Divide texto em chunks de tamanho fixo (idêntico ao gerador)."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if len(chunk) > MIN_CHUNK_LEN:
            chunks.append(chunk)
    return chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Carregar dataset uma vez
    dataset = get_dataset(DATASET_NAME)
    train_ds = dataset["train"]

    # Índice rápido blob_path -> text
    blob_to_text = {}
    for item in train_ds:
        blob = item.get("blob_path")
        text = item.get("text")
        if blob and text:
            blob_to_text[blob] = text

    chunks_cache = {}

    total_lines = 0
    enriched_lines = 0
    skipped_out_of_range = 0

    with INPUT_FILE.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Enriquecendo", unit="linha"):
            total_lines += 1
            raw = line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                skipped_out_of_range += 1
                continue

            doc_id = obj.get("doc_id") or obj.get("blob_path")
            chunk_idx = obj.get("chunk_idx")

            if doc_id is None or chunk_idx is None:
                skipped_out_of_range += 1
                continue

            # Cache de chunks por documento
            if doc_id not in chunks_cache:
                text = blob_to_text.get(doc_id)
                if text is None:
                    skipped_out_of_range += 1
                    continue
                chunks_cache[doc_id] = split_into_chunks(text, CHUNK_SIZE)

            chunks = chunks_cache[doc_id]

            if not isinstance(chunk_idx, int) or chunk_idx < 0 or chunk_idx >= len(chunks):
                skipped_out_of_range += 1
                continue

            chunk_text = chunks[chunk_idx]
            obj["chunk_text"] = chunk_text
            obj["chunk_len"] = len(chunk_text)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            enriched_lines += 1

    total_time = time.time() - start_time

    report = {
        "total_lines": total_lines,
        "enriched_lines": enriched_lines,
        "skipped_out_of_range": skipped_out_of_range,
        "docs_cached": len(chunks_cache),
        "tempo_total": total_time,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("ENRIQUECIMENTO SNAPSHOT_V1.JSONL")
    print("=" * 70)
    print(f"Total linhas: {total_lines}")
    print(f"Enriquecidas: {enriched_lines}")
    print(f"Puladas (out of range/erro): {skipped_out_of_range}")
    print(f"Docs em cache: {len(chunks_cache)}")
    print(f"Tempo total: {total_time:.1f}s")
    print(f"Saída: {OUTPUT_FILE}")
    print(f"Relatório: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
