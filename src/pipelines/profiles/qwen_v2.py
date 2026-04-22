"""Qwen v2 profile defaults.

This file is intentionally profile-scoped so future profiles can be added
without changing generic pipeline code.
"""

from datetime import datetime

import torch

from src.core.config import env_bool, env_float, env_int, env_str
from src.core.paths import PROJECT_ROOT


SEED = 42
TOP_K_METRICS = [1, 5, 10]

EXP_V2 = PROJECT_ROOT / "experiments" / "exp_v2_40k"
EXP_RERANK = PROJECT_ROOT / "experiments" / "exp_v2_40k_reranker_ft"
EXP_ENERGY = PROJECT_ROOT / "experiments" / "exp_energy_eval_qwen"


def train_encoder_config():
    train_file = EXP_V2 / "train_pos_v2_train.jsonl"
    model_dir = EXP_V2 / "models" / "qwen3_embedding_0_6b_ft_v2"

    return {
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "base_model": env_str("BASE_ENCODER_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        "train_file": train_file,
        "output_dir": model_dir,
        "train_config_file": EXP_V2 / "train_config_qwen_v2.json",
        "train_summary_file": EXP_V2 / "train_summary_qwen_v2.json",
        "epochs": env_int("TRAIN_EPOCHS", 1),
        "batch_size": env_int("TRAIN_BATCH_SIZE", 8),
        "warmup_ratio": env_float("TRAIN_WARMUP_RATIO", 0.05),
        "cached_mini_batch_size": env_int("CACHED_MINI_BATCH_SIZE", 8),
        "shuffle": True,
        "require_cuda": True,
        "loss_function": "CachedMultipleNegativesRankingLoss",
        "trust_remote_code": True,
        "model_kwargs": {
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        },
        "tokenizer_kwargs": {"fix_mistral_regex": True},
        "gradient_checkpointing": True,
    }


def eval_retrieval_config():
    return {
        "seed": SEED,
        "top_k_metrics": TOP_K_METRICS,
        "batch_size_encode": env_int("BATCH_SIZE_ENCODE", 64),
        "batch_size_rerank": env_int("BATCH_SIZE_RERANK", 32),
        "top_k_retrieve": env_int("TOP_K_RETRIEVE", 100),
        "enable_cross_encoder": env_bool("ENABLE_CROSS_ENCODER", True),
        "require_cuda": True,
        "test_file": EXP_V2 / "train_pos_v2_test.jsonl",
        "encoder_model": EXP_V2 / "models" / "qwen3_embedding_0_6b_ft_v2",
        "reranker_model": env_str("RERANKER_MODEL", str(EXP_RERANK / "models" / "reranker_ft")),
        "results_legacy": EXP_V2 / "results_qwen_ft_v2.json",
        "metrics_legacy": EXP_V2 / "metrics_qwen_ft_v2.md",
        "results_no_cross": EXP_V2 / "results_qwen_ft_v2_no_cross.json",
        "results_cross": EXP_V2 / "results_qwen_ft_v2_cross.json",
        "results_compare": EXP_V2 / "results_qwen_ft_v2_compare.json",
        "metrics_compare": EXP_V2 / "metrics_qwen_ft_v2_compare.md",
        "tokenizer_kwargs": {"fix_mistral_regex": True},
    }


def eval_mcq_config():
    return {
        "seed": SEED,
        "top_k_metrics": TOP_K_METRICS,
        "batch_size_encode": env_int("BATCH_SIZE_ENCODE", 64),
        "batch_size_rerank": env_int("BATCH_SIZE_RERANK", 32),
        "top_k_retrieve": env_int("TOP_K_RETRIEVE", 100),
        "top_n_contexts_for_decoder": env_int("TOP_N_CONTEXTS_FOR_DECODER", 3),
        "context_max_chars": env_int("CONTEXT_MAX_CHARS", 2200),
        "decoder_max_new_tokens": env_int("DECODER_MAX_NEW_TOKENS", 8),
        "eval_max_rows": env_int("EVAL_MAX_ROWS", 0),
        "require_cuda": True,
        "dataset_file": PROJECT_ROOT / "datasets" / "energy_eval" / "train-00000-of-00001.parquet",
        "encoder_model": EXP_V2 / "models" / "qwen3_embedding_0_6b_ft_v2",
        "reranker_model": env_str("RERANKER_MODEL", str(EXP_RERANK / "models" / "reranker_ft")),
        "decoder_model": env_str("DECODER_MODEL_NAME", "Qwen/Qwen3.5-9B"),
        "results_no_cross": EXP_ENERGY / "results_energy_eval_qwen_no_cross.json",
        "results_cross": EXP_ENERGY / "results_energy_eval_qwen_cross.json",
        "results_compare": EXP_ENERGY / "results_energy_eval_qwen_compare.json",
        "metrics_compare": EXP_ENERGY / "metrics_energy_eval_qwen_compare.md",
        "predictions_no_cross": EXP_ENERGY / "predictions_energy_eval_qwen_no_cross.jsonl",
        "predictions_cross": EXP_ENERGY / "predictions_energy_eval_qwen_cross.jsonl",
        "output_dir": EXP_ENERGY,
        "tokenizer_kwargs": {"fix_mistral_regex": True},
    }
