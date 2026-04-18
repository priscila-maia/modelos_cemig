#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treino Qwen3-Embedding-0.6B no dataset v2
Mesmo dataset e mesma loss do treino E5 v2.
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

SEED = 42

BASE_DIR = Path(__file__).parent
EXP_DIR = BASE_DIR / "experiments" / "exp_v2_40k"
TRAIN_FILE = EXP_DIR / "train_pos_v2_train.jsonl"
MODEL_DIR = EXP_DIR / "models" / "qwen3_embedding_0_6b_ft_v2"
TRAIN_SUMMARY_FILE = EXP_DIR / "train_summary_qwen_v2.json"
TRAIN_CONFIG_FILE = EXP_DIR / "train_config_qwen_v2.json"

BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("TRAIN_EPOCHS", "1"))
WARMUP_RATIO = float(os.getenv("TRAIN_WARMUP_RATIO", "0.05"))
CACHED_MINI_BATCH_SIZE = int(os.getenv("CACHED_MINI_BATCH_SIZE", "8"))

CONFIG = {
    "base_model": "Qwen/Qwen3-Embedding-0.6B",
    "train_file": str(TRAIN_FILE),
    "output_dir": str(MODEL_DIR),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "warmup_ratio": WARMUP_RATIO,
    "cached_mini_batch_size": CACHED_MINI_BATCH_SIZE,
    "seed": SEED,
    "loss_function": "CachedMultipleNegativesRankingLoss",
    "shuffle": True,
    "device_required": "cuda",
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


def load_qwen_model(device: str) -> SentenceTransformer:
    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }

    # sentence-transformers >=5 aceita model_kwargs e trust_remote_code.
    return SentenceTransformer(
        CONFIG["base_model"],
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"fix_mistral_regex": True},
    )


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {TRAIN_FILE}")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA não disponível. Este treino foi configurado para usar GPU.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    start = time.time()

    with TRAIN_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    device = "cuda"
    model = load_qwen_model(device)
    if hasattr(model, "_first_module"):
        first = model._first_module()
        auto_model = getattr(first, "auto_model", None)
        if auto_model is not None and hasattr(auto_model, "gradient_checkpointing_enable"):
            auto_model.gradient_checkpointing_enable()

    train_examples = load_train_data(TRAIN_FILE)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=CONFIG["shuffle"],
        batch_size=CONFIG["batch_size"],
    )

    # Mesma loss do treino E5.
    train_loss = losses.CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=CACHED_MINI_BATCH_SIZE,
    )

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
        "cuda_device": torch.cuda.get_device_name(0),
    }

    with TRAIN_SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("TREINO QWEN3-EMBEDDING-0.6B V2 CONCLUÍDO")
    print("=" * 70)
    print(f"Pares de treino: {len(train_examples)}")
    print(f"Tempo total: {total_time:.1f}s")
    print(f"Modelo: {MODEL_DIR}")
    print(f"Summary: {TRAIN_SUMMARY_FILE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
