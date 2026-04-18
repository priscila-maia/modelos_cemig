#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerar pares positivos de treino a partir de snapshot_v1_filtrado.jsonl
"""

import json
import time
from pathlib import Path

from tqdm import tqdm


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
INPUT_FILE = DATA_DIR / "snapshot_v1_filtrado.jsonl"
OUTPUT_FILE = DATA_DIR / "train_pos_v1.jsonl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "build_pairs_v1_report.json"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    input_lines = 0
    pairs_written = 0

    with INPUT_FILE.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Gerando pares", unit="linha"):
            raw = line.strip()
            if not raw:
                continue
            input_lines += 1

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            doc_id = obj.get("doc_id") or obj.get("blob_path")
            chunk_idx = obj.get("chunk_idx")
            chunk_text = obj.get("chunk_text")
            chunk_len = obj.get("chunk_len")

            if doc_id is None or chunk_idx is None or chunk_text is None or chunk_len is None:
                continue

            gold_1 = obj.get("pergunta_gold_1")
            gold_2 = obj.get("pergunta_gold_2")

            if isinstance(gold_1, str) and gold_1.strip():
                out_1 = {
                    "doc_id": doc_id,
                    "chunk_idx": int(chunk_idx),
                    "query": gold_1.strip(),
                    "context": chunk_text,
                    "label": 1,
                    "source": "gold_1",
                    "chunk_len": int(chunk_len),
                }
                fout.write(json.dumps(out_1, ensure_ascii=False) + "\n")
                pairs_written += 1

            if isinstance(gold_2, str) and gold_2.strip():
                out_2 = {
                    "doc_id": doc_id,
                    "chunk_idx": int(chunk_idx),
                    "query": gold_2.strip(),
                    "context": chunk_text,
                    "label": 1,
                    "source": "gold_2",
                    "chunk_len": int(chunk_len),
                }
                fout.write(json.dumps(out_2, ensure_ascii=False) + "\n")
                pairs_written += 1

    tempo_total = time.time() - start_time

    report = {
        "input_lines": input_lines,
        "pairs_written": pairs_written,
        "tempo_total": tempo_total,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("BUILD TRAIN PAIRS NOVO")
    print("=" * 70)
    print(f"Input lines: {input_lines}")
    print(f"Pairs written: {pairs_written}")
    print(f"Tempo total: {tempo_total:.1f}s")
    print(f"Saída: {OUTPUT_FILE}")
    print(f"Relatório: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
