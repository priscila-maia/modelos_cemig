#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação do snapshot snapshot_v1.jsonl (40k)
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
INPUT_FILE = DATA_DIR / "snapshot_v1.jsonl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "validate_v1_report.json"

REQUIRED_QUESTIONS = [
    "pergunta_gold_1",
    "pergunta_gold_2",
    "pergunta_hard_1",
    "pergunta_hard_2",
]

LEAK_PATTERNS = [
    "Documento:",
    "Responda SOMENTE em JSON",
    "Responda somente em JSON",
    "Responda em JSON",
    "Somente em JSON",
    "FORMATO JSON",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    valid_json_lines = 0
    invalid_json_lines = 0
    empty_lines = 0

    missing_keys_count = {
        "doc_id": 0,
        "blob_path": 0,
        "chunk_idx": 0,
        **{k: 0 for k in REQUIRED_QUESTIONS},
    }

    empty_question_count = 0
    short_question_count = 0
    leaked_prompt_count = 0

    exemplos_de_erros = []
    exemplos_de_leak = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1
            raw = line.rstrip("\n")

            if not raw.strip():
                empty_lines += 1
                continue

            try:
                obj = json.loads(raw)
                valid_json_lines += 1
            except json.JSONDecodeError as e:
                invalid_json_lines += 1
                if len(exemplos_de_erros) < 5:
                    exemplos_de_erros.append(
                        {"line": line_no, "error": str(e), "content": raw[:300]}
                    )
                continue

            # doc_id ou blob_path
            has_doc_id = "doc_id" in obj and str(obj.get("doc_id", "")).strip() != ""
            has_blob_path = "blob_path" in obj and str(obj.get("blob_path", "")).strip() != ""
            if not has_doc_id:
                missing_keys_count["doc_id"] += 1
            if not has_blob_path:
                missing_keys_count["blob_path"] += 1

            # chunk_idx
            if "chunk_idx" not in obj:
                missing_keys_count["chunk_idx"] += 1

            # perguntas
            for q_key in REQUIRED_QUESTIONS:
                q_val = obj.get(q_key, "")
                if not isinstance(q_val, str) or not q_val.strip():
                    missing_keys_count[q_key] += 1
                    empty_question_count += 1
                else:
                    if len(q_val.strip()) < 10:
                        short_question_count += 1

            # leak
            content_to_check = " ".join(
                [
                    str(obj.get(k, ""))
                    for k in REQUIRED_QUESTIONS
                    if isinstance(obj.get(k, ""), str)
                ]
            )
            if any(pat in content_to_check for pat in LEAK_PATTERNS):
                leaked_prompt_count += 1
                if len(exemplos_de_leak) < 5:
                    exemplos_de_leak.append(
                        {"line": line_no, "snippet": content_to_check[:300]}
                    )

    report = {
        "total_lines": total_lines,
        "valid_json_lines": valid_json_lines,
        "invalid_json_lines": invalid_json_lines,
        "empty_lines": empty_lines,
        "missing_keys_count": missing_keys_count,
        "empty_question_count": empty_question_count,
        "short_question_count": short_question_count,
        "leaked_prompt_count": leaked_prompt_count,
        "exemplos_de_erros": exemplos_de_erros,
        "exemplos_de_leak": exemplos_de_leak,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("VALIDAÇÃO SNAPSHOT_V1.JSONL")
    print("=" * 70)
    print(f"Total linhas: {total_lines}")
    print(f"JSON válidos: {valid_json_lines}")
    print(f"JSON inválidos: {invalid_json_lines}")
    print(f"Linhas vazias: {empty_lines}")
    print(f"Perguntas vazias: {empty_question_count}")
    print(f"Perguntas curtas (<10): {short_question_count}")
    print(f"Prompt vazado: {leaked_prompt_count}")
    print(f"Relatório: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
