#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtro conservador para snapshot_v1_enriquecido.jsonl
"""

import json
import time
from pathlib import Path

from tqdm import tqdm


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
INPUT_FILE = DATA_DIR / "snapshot_v1_enriquecido.jsonl"
OUTPUT_FILE = DATA_DIR / "snapshot_v1_filtrado.jsonl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "filter_v1_report.json"

QUESTIONS = [
    "pergunta_gold_1",
    "pergunta_gold_2",
    "pergunta_hard_1",
    "pergunta_hard_2",
]

GENERIC_PATTERNS = [
    "explique",
    "resuma",
    "fale sobre",
    "o que é",
    "defina",
    "quais são",
    "como funciona",
    "descreva",
    "dê detalhes",
    "me explique",
]

MIN_CHUNK_LEN = 150
MIN_QUESTION_LEN = 25


def is_generic_question(q: str) -> bool:
    q_norm = q.strip().lower()
    return any(pat in q_norm for pat in GENERIC_PATTERNS)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    total_lines = 0
    kept_lines = 0
    removed_lines = 0

    breakdown = {
        "removed_small_chunk": 0,
        "removed_short_question": 0,
        "removed_generic_question": 0,
    }

    exemplos_removidos = []

    with INPUT_FILE.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(tqdm(fin, desc="Filtrando", unit="linha"), start=1):
            total_lines += 1
            raw = line.strip()
            if not raw:
                removed_lines += 1
                breakdown["removed_short_question"] += 1
                if len(exemplos_removidos) < 5:
                    exemplos_removidos.append({"line": line_no, "motivo": "linha_vazia"})
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                removed_lines += 1
                breakdown["removed_short_question"] += 1
                if len(exemplos_removidos) < 5:
                    exemplos_removidos.append({"line": line_no, "motivo": "json_invalido"})
                continue

            chunk_len = obj.get("chunk_len")
            if not isinstance(chunk_len, int) or chunk_len < MIN_CHUNK_LEN:
                removed_lines += 1
                breakdown["removed_small_chunk"] += 1
                if len(exemplos_removidos) < 5:
                    exemplos_removidos.append({"line": line_no, "motivo": "chunk_curto"})
                continue

            # Verificar perguntas
            remove_due_short = False
            remove_due_generic = False

            for q_key in QUESTIONS:
                q_val = obj.get(q_key, "")
                if not isinstance(q_val, str) or len(q_val.strip()) < MIN_QUESTION_LEN:
                    remove_due_short = True
                    break
                if is_generic_question(q_val):
                    remove_due_generic = True
                    break

            if remove_due_short:
                removed_lines += 1
                breakdown["removed_short_question"] += 1
                if len(exemplos_removidos) < 5:
                    exemplos_removidos.append({"line": line_no, "motivo": "pergunta_curta"})
                continue

            if remove_due_generic:
                removed_lines += 1
                breakdown["removed_generic_question"] += 1
                if len(exemplos_removidos) < 5:
                    exemplos_removidos.append({"line": line_no, "motivo": "pergunta_generica"})
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept_lines += 1

    tempo_total = time.time() - start_time
    percent_removed = (removed_lines / total_lines * 100) if total_lines > 0 else 0

    report = {
        "total_lines": total_lines,
        "kept_lines": kept_lines,
        "removed_lines": removed_lines,
        "percent_removed": percent_removed,
        "breakdown_por_motivo": breakdown,
        "exemplos_removidos": exemplos_removidos,
        "tempo_total": tempo_total,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("FILTRO SNAPSHOT_V1.JSONL")
    print("=" * 70)
    print(f"Total linhas: {total_lines}")
    print(f"Mantidos: {kept_lines}")
    print(f"Removidos: {removed_lines} ({percent_removed:.2f}%)")
    print(f"Motivos: {breakdown}")
    print(f"Relatório: {REPORT_FILE}")
    print(f"Saída: {OUTPUT_FILE}")
    print(f"Tempo total: {tempo_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
