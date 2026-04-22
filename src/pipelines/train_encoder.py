"""Generic sentence-transformer encoder training pipeline."""

import os
import time

import torch
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

from src.core.cache import setup_hf_cache_dirs
from src.core.io import write_json
from src.core.seed import set_all_seeds
from src.data.jsonl import load_query_context_pairs
from src.pipelines.profiles import get_profile_module
from src.retrieval.encoder import load_sentence_encoder


def run_train_encoder(profile_name: str):
    profile = get_profile_module(profile_name)
    cfg = profile.train_encoder_config()

    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]
    train_cfg_file = cfg["train_config_file"]
    train_summary_file = cfg["train_summary_file"]

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if cfg.get("require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training profile.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    setup_hf_cache_dirs()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_file.parent.mkdir(parents=True, exist_ok=True)

    set_all_seeds(cfg["seed"])
    start = time.time()

    config_payload = {
        "base_model": cfg["base_model"],
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "warmup_ratio": cfg["warmup_ratio"],
        "cached_mini_batch_size": cfg["cached_mini_batch_size"],
        "seed": cfg["seed"],
        "loss_function": cfg["loss_function"],
        "shuffle": cfg["shuffle"],
        "timestamp": cfg["timestamp"],
        "profile": profile_name,
    }
    write_json(train_cfg_file, config_payload)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sentence_encoder(
        cfg["base_model"],
        device=device,
        trust_remote_code=cfg.get("trust_remote_code", False),
        model_kwargs=cfg.get("model_kwargs"),
        tokenizer_kwargs=cfg.get("tokenizer_kwargs"),
    )

    if cfg.get("gradient_checkpointing", False):
        first = model._first_module() if hasattr(model, "_first_module") else None
        auto_model = getattr(first, "auto_model", None)
        if auto_model is not None and hasattr(auto_model, "gradient_checkpointing_enable"):
            auto_model.gradient_checkpointing_enable()

    pairs = load_query_context_pairs(train_file)
    train_examples = [InputExample(texts=[p["query"], p["context"]]) for p in pairs]

    dataloader = DataLoader(
        train_examples,
        shuffle=cfg["shuffle"],
        batch_size=cfg["batch_size"],
    )

    train_loss = losses.CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=cfg["cached_mini_batch_size"],
    )

    num_train_steps = len(dataloader) * cfg["epochs"]
    warmup_steps = int(num_train_steps * cfg["warmup_ratio"])

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=cfg["epochs"],
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
        save_best_model=True,
    )

    total_time = time.time() - start
    summary = {
        **config_payload,
        "num_train_examples": len(train_examples),
        "num_train_steps": num_train_steps,
        "warmup_steps": warmup_steps,
        "total_time_sec": total_time,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    write_json(train_summary_file, summary)

    print("=" * 70)
    print("TRAIN ENCODER COMPLETED")
    print("=" * 70)
    print(f"Profile: {profile_name}")
    print(f"Pairs: {len(train_examples)}")
    print(f"Time: {total_time:.1f}s")
    print(f"Model: {output_dir}")
    print(f"Summary: {train_summary_file}")
    print("=" * 70)
