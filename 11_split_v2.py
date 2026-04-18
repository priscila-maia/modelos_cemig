#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split 90/10 do train_pos_v2.jsonl (seed=42)
"""

import json
import random
import time
from pathlib import Path
from datetime import datetime

SEED = 42
TRAIN_RATIO = 0.9
TEST_RATIO = 0.1

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
INPUT_FILE = DATA_DIR / "train_pos_v2.jsonl"
EXP_DIR = BASE_DIR / "experiments" / "exp_v2_40k"
TRAIN_FILE = EXP_DIR / "train_pos_v2_train.jsonl"
TEST_FILE = EXP_DIR / "train_pos_v2_test.jsonl"
CONFIG_FILE = EXP_DIR / "split_config_v2.json"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    start = time.time()

    all_pairs = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                all_pairs.append(obj)
            except json.JSONDecodeError:
                continue

    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * TRAIN_RATIO)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]

    with TRAIN_FILE.open("w", encoding="utf-8") as f:
        for obj in train_pairs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with TEST_FILE.open("w", encoding="utf-8") as f:
        for obj in test_pairs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    config = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "test_ratio": TEST_RATIO,
        "input_file": str(INPUT_FILE),
        "train_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "tempo_total": time.time() - start,
    }

    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("SPLIT DATASET V2 (90/10)")
    print("=" * 70)
    print(f"Total: {len(all_pairs)}")
    print(f"Treino: {len(train_pairs)}")
    print(f"Teste: {len(test_pairs)}")
    print(f"Config: {CONFIG_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
