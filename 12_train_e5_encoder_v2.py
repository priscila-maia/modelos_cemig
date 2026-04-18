#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treino E5-Large no dataset v2
"""

import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

SEED = 42

BASE_DIR = Path(__file__).parent
EXP_DIR = BASE_DIR / "experiments" / "exp_v2_40k"
TRAIN_FILE = EXP_DIR / "train_pos_v2_train.jsonl"
MODEL_DIR = EXP_DIR / "models" / "e5_large_ft_v2"
TRAIN_SUMMARY_FILE = EXP_DIR / "train_summary.json"
TRAIN_CONFIG_FILE = EXP_DIR / "train_config_v2.json"

CONFIG = {
    "base_model": "intfloat/multilingual-e5-large",
    "train_file": str(TRAIN_FILE),
    "output_dir": str(MODEL_DIR),
    "epochs": 1,
    "batch_size": 32,
    "warmup_ratio": 0.05,
    "seed": SEED,
    "loss_function": "CachedMultipleNegativesRankingLoss",
    "shuffle": True,
    "timestamp": datetime.now().isoformat(),
}


def load_train_data(file_path: Path):
    examples = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            query = obj.get("query", "")
            context = obj.get("context", "")
            if query and context:
                examples.append(InputExample(texts=[query, context]))
    return examples


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {TRAIN_FILE}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    start = time.time()

    with TRAIN_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CONFIG["base_model"], device=device)

    train_examples = load_train_data(TRAIN_FILE)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=CONFIG["shuffle"],
        batch_size=CONFIG["batch_size"],
    )

    train_loss = losses.CachedMultipleNegativesRankingLoss(model)

    num_train_steps = len(train_dataloader) * CONFIG["epochs"]
    warmup_steps = int(num_train_steps * CONFIG["warmup_ratio"])

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=CONFIG["epochs"],
        warmup_steps=warmup_steps,
        output_path=str(MODEL_DIR),
        show_progress_bar=True,
        save_best_model=True,
    )

    total_time = time.time() - start

    summary = {
        **CONFIG,
        "num_train_examples": len(train_examples),
        "num_train_steps": num_train_steps,
        "warmup_steps": warmup_steps,
        "total_time_sec": total_time,
    }

    with TRAIN_SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("TREINO E5-LARGE V2 CONCLUÍDO")
    print("=" * 70)
    print(f"Pares de treino: {len(train_examples)}")
    print(f"Tempo total: {total_time:.1f}s")
    print(f"Modelo: {MODEL_DIR}")
    print(f"Summary: {TRAIN_SUMMARY_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
