#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treino de Cross-Encoder (reranker) fine-tuned no dataset v2.
"""

import json
import time
import random
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

SEED = 42

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"
EXP_DIR = BASE_DIR / "experiments" / "exp_v2_40k_reranker_ft"
MODEL_DIR = EXP_DIR / "models" / "reranker_ft"
TRAIN_FILE = DATA_DIR / "train_cross_encoder.jsonl"
TRAIN_CONFIG_FILE = EXP_DIR / "train_config.json"
TRAIN_LOG_FILE = EXP_DIR / "train_log.txt"

CONFIG = {
    "base_model": "BAAI/bge-reranker-v2-m3",
    "train_file": str(TRAIN_FILE),
    "output_dir": str(MODEL_DIR),
    "epochs": 1,
    "batch_size": 8,
    "max_length": 512,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.05,
    "seed": SEED,
    "loss": "CrossEntropyLoss",
    "timestamp": datetime.now().isoformat(),
}


def setup_logger(log_path: Path):
    logger = logging.getLogger("reranker_ft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


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
            label = obj.get("label")
            if query and context and label in (0, 1):
                examples.append(InputExample(texts=[query, context], label=float(label)))
    return examples


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {TRAIN_FILE}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(TRAIN_LOG_FILE)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    start = time.time()

    with TRAIN_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    logger.info("Carregando modelo base: %s", CONFIG["base_model"])
    model = CrossEncoder(CONFIG["base_model"], max_length=CONFIG["max_length"])

    logger.info("Carregando dataset: %s", TRAIN_FILE)
    train_examples = load_train_data(TRAIN_FILE)
    logger.info("Total de exemplos: %d", len(train_examples))

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CONFIG["batch_size"],
    )

    num_train_steps = len(train_dataloader) * CONFIG["epochs"]
    warmup_steps = int(num_train_steps * CONFIG["warmup_ratio"])

    logger.info("Iniciando treino...")
    logger.info("Batch size: %d", CONFIG["batch_size"])
    logger.info("Epochs: %d", CONFIG["epochs"])
    logger.info("Warmup steps: %d", warmup_steps)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=CONFIG["epochs"],
        warmup_steps=warmup_steps,
        optimizer_params={"lr": CONFIG["learning_rate"]},
        show_progress_bar=True,
        output_path=str(MODEL_DIR),
    )

    # Garantir salvamento explícito
    model.save(str(MODEL_DIR))

    # Checagem automática
    has_config = (MODEL_DIR / "config.json").exists()
    has_weights = (MODEL_DIR / "model.safetensors").exists() or (MODEL_DIR / "pytorch_model.bin").exists()
    if not (has_config and has_weights):
        raise RuntimeError("Falha ao salvar modelo: config.json ou pesos não encontrados")

    total_time = time.time() - start

    logger.info("Treino concluído")
    logger.info("Tempo total: %.1fs", total_time)
    logger.info("Modelo salvo em: %s", MODEL_DIR)

    print("=" * 70)
    print("TREINO RERANKER FT CONCLUÍDO")
    print("=" * 70)
    print(f"Tempo total: {total_time:.1f}s")
    print(f"Passos: {num_train_steps}")
    print(f"Modelo: {MODEL_DIR}")
    print(f"Log: {TRAIN_LOG_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
