"""Pipeline for retrieval + decoder evaluation on MCQ datasets."""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.core.cache import resolve_hf_cache_dir, setup_hf_cache_dirs
from src.core.io import write_json, write_jsonl, write_text
from src.core.metrics import build_rank_metrics, sanity_test_mrr
from src.core.seed import set_all_seeds
from src.data.energy_eval import build_context_corpus, load_energy_eval_samples
from src.generation.decoder import decode_choice, load_causal_decoder
from src.generation.prompts import build_mcq_prompt
from src.pipelines.profiles import get_profile_module
from src.retrieval.encoder import (
    encode_texts,
    load_sentence_encoder,
    rank_indices_by_cosine,
    topk_indices_by_cosine,
)
from src.retrieval.rerank import load_cross_encoder, rerank_indices


def _decoder_metrics(predictions):
    total = len(predictions)
    total_correct = sum(1 for p in predictions if p["correct"])
    total_parsed = sum(1 for p in predictions if p["predicted"] != "N/A")

    subset = [p for p in predictions if p["gt_context_in_prompt"]]
    subset_acc = sum(1 for p in subset if p["correct"]) / len(subset) if subset else 0.0

    return {
        "accuracy": total_correct / total,
        "parsed_rate": total_parsed / total,
        "accuracy_when_gt_context_in_prompt": subset_acc,
        "num_items": total,
        "num_items_gt_context_in_prompt": len(subset),
    }


def _eval_retrieval_no_cross(query_embeddings, context_embeddings, samples, top_n_contexts_for_decoder):
    rankings = []
    retrieved = []
    for i in tqdm(range(len(samples)), desc="Retrieval no cross"):
        target_idx = samples[i]["target_context_idx"]
        sorted_indices = rank_indices_by_cosine(query_embeddings[i], context_embeddings)
        rank = int(np.where(sorted_indices == target_idx)[0][0] + 1)
        rankings.append(rank)
        retrieved.append(sorted_indices[:top_n_contexts_for_decoder].tolist())
    return rankings, retrieved


def _eval_retrieval_with_cross(
    query_embeddings,
    context_embeddings,
    samples,
    questions,
    contexts,
    reranker,
    top_k_retrieve,
    top_n_contexts_for_decoder,
    batch_size_rerank,
):
    rankings = []
    retrieved = []
    for i in tqdm(range(len(samples)), desc="Retrieval with cross"):
        target_idx = samples[i]["target_context_idx"]
        top_k = min(top_k_retrieve, len(contexts))
        candidate_indices = topk_indices_by_cosine(query_embeddings[i], context_embeddings, k=top_k)
        reranked_indices = rerank_indices(
            query=questions[i],
            contexts=contexts,
            candidate_indices=candidate_indices,
            reranker=reranker,
            batch_size=batch_size_rerank,
        )
        rank_arr = np.where(reranked_indices == target_idx)[0]
        rank = int(rank_arr[0] + 1) if len(rank_arr) > 0 else top_k_retrieve + 1
        rankings.append(rank)
        retrieved.append(reranked_indices[:top_n_contexts_for_decoder].tolist())
    return rankings, retrieved


def _eval_decoder(samples, corpus_contexts, retrieved_context_indices, tokenizer, model, context_max_chars, max_new_tokens, mode_name):
    predictions = []
    for i in tqdm(range(len(samples)), desc=f"Decoder {mode_name}"):
        sample = samples[i]
        idxs = retrieved_context_indices[i]
        contexts = [corpus_contexts[idx] for idx in idxs]
        prompt = build_mcq_prompt(sample["question"], sample["choices"], contexts, context_max_chars=context_max_chars)
        pred_choice, raw_generation = decode_choice(prompt, tokenizer, model, max_new_tokens=max_new_tokens)
        answer_key = sample["answerKey"]

        predictions.append(
            {
                "id": sample["id"],
                "question_number": sample["question_number"],
                "answer_key": answer_key,
                "predicted": pred_choice,
                "correct": bool(pred_choice == answer_key),
                "gt_context_in_prompt": bool(sample["target_context_idx"] in idxs),
                "retrieved_context_indices": idxs,
                "decoder_raw": raw_generation,
            }
        )

    return predictions


def run_eval_mcq(profile_name: str):
    profile = get_profile_module(profile_name)
    cfg = profile.eval_mcq_config()

    dataset_file = Path(cfg["dataset_file"])
    encoder_model = Path(cfg["encoder_model"])
    reranker_model = Path(cfg["reranker_model"])

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    if not encoder_model.exists():
        raise FileNotFoundError(f"Encoder model not found: {encoder_model}")
    if not reranker_model.exists():
        raise FileNotFoundError(f"Reranker model not found: {reranker_model}")
    if cfg.get("require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation profile.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(cfg["seed"])
    setup_hf_cache_dirs()
    sanity_test_mrr(cfg["top_k_metrics"])

    samples = load_energy_eval_samples(dataset_file, max_rows=cfg["eval_max_rows"])
    if not samples:
        raise RuntimeError("No valid samples found in energy eval dataset.")
    corpus_contexts, samples = build_context_corpus(samples)
    questions = [s["question"] for s in samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_sentence_encoder(
        str(encoder_model),
        device=device,
        tokenizer_kwargs=cfg.get("tokenizer_kwargs"),
    )

    print("Encoding corpus contexts...")
    context_embeddings = encode_texts(encoder, corpus_contexts, batch_size=cfg["batch_size_encode"])
    print("Encoding questions...")
    query_embeddings = encode_texts(encoder, questions, batch_size=cfg["batch_size_encode"])

    cache_dir = str(resolve_hf_cache_dir() / "transformers")
    tokenizer, decoder = load_causal_decoder(cfg["decoder_model"], cache_dir=cache_dir)

    start_no_cross = time.time()
    rankings_no_cross, retrieved_no_cross = _eval_retrieval_no_cross(
        query_embeddings,
        context_embeddings,
        samples,
        cfg["top_n_contexts_for_decoder"],
    )
    preds_no_cross = _eval_decoder(
        samples,
        corpus_contexts,
        retrieved_no_cross,
        tokenizer,
        decoder,
        context_max_chars=cfg["context_max_chars"],
        max_new_tokens=cfg["decoder_max_new_tokens"],
        mode_name="no_cross",
    )
    no_cross_time = time.time() - start_no_cross

    retrieval_no_cross = build_rank_metrics(rankings_no_cross, cfg["top_k_metrics"])
    decoder_no_cross = _decoder_metrics(preds_no_cross)

    result_no_cross = {
        "profile": profile_name,
        "mode": "no_cross_encoder",
        "dataset": str(dataset_file),
        "encoder_model": str(encoder_model),
        "decoder_model": cfg["decoder_model"],
        "timestamp": datetime.now().isoformat(),
        "seed": cfg["seed"],
        "num_samples": len(samples),
        "num_unique_contexts": len(corpus_contexts),
        "top_k_metrics": cfg["top_k_metrics"],
        "top_n_contexts_for_decoder": cfg["top_n_contexts_for_decoder"],
        "total_time_sec": no_cross_time,
        "retrieval_metrics": retrieval_no_cross,
        "decoder_metrics": decoder_no_cross,
    }
    write_json(Path(cfg["results_no_cross"]), result_no_cross)
    write_jsonl(Path(cfg["predictions_no_cross"]), preds_no_cross)

    start_cross = time.time()
    reranker = load_cross_encoder(str(reranker_model), device=device)
    rankings_cross, retrieved_cross = _eval_retrieval_with_cross(
        query_embeddings,
        context_embeddings,
        samples,
        questions,
        corpus_contexts,
        reranker,
        cfg["top_k_retrieve"],
        cfg["top_n_contexts_for_decoder"],
        cfg["batch_size_rerank"],
    )
    preds_cross = _eval_decoder(
        samples,
        corpus_contexts,
        retrieved_cross,
        tokenizer,
        decoder,
        context_max_chars=cfg["context_max_chars"],
        max_new_tokens=cfg["decoder_max_new_tokens"],
        mode_name="with_cross",
    )
    cross_time = time.time() - start_cross

    retrieval_cross = build_rank_metrics(rankings_cross, cfg["top_k_metrics"])
    decoder_cross = _decoder_metrics(preds_cross)

    result_cross = {
        "profile": profile_name,
        "mode": "with_cross_encoder",
        "dataset": str(dataset_file),
        "encoder_model": str(encoder_model),
        "reranker_model": str(reranker_model),
        "decoder_model": cfg["decoder_model"],
        "top_k_retrieve": cfg["top_k_retrieve"],
        "timestamp": datetime.now().isoformat(),
        "seed": cfg["seed"],
        "num_samples": len(samples),
        "num_unique_contexts": len(corpus_contexts),
        "top_k_metrics": cfg["top_k_metrics"],
        "top_n_contexts_for_decoder": cfg["top_n_contexts_for_decoder"],
        "total_time_sec": cross_time,
        "retrieval_metrics": retrieval_cross,
        "decoder_metrics": decoder_cross,
    }
    write_json(Path(cfg["results_cross"]), result_cross)
    write_jsonl(Path(cfg["predictions_cross"]), preds_cross)

    compare_payload = {
        "profile": profile_name,
        "dataset": str(dataset_file),
        "encoder_model": str(encoder_model),
        "reranker_model": str(reranker_model),
        "decoder_model": cfg["decoder_model"],
        "timestamp": datetime.now().isoformat(),
        "seed": cfg["seed"],
        "num_samples": len(samples),
        "num_unique_contexts": len(corpus_contexts),
        "no_cross": result_no_cross,
        "with_cross": result_cross,
    }
    write_json(Path(cfg["results_compare"]), compare_payload)

    write_text(
        Path(cfg["metrics_compare"]),
        f"""# Energy Eval - Retrieval + Decoder

**Profile:** {profile_name}  
**Dataset:** `{dataset_file}`  
**Encoder:** `{encoder_model}`  
**Reranker:** `{reranker_model}`  
**Decoder:** `{cfg['decoder_model']}`  
**Samples:** {len(samples)}  
**Unique contexts:** {len(corpus_contexts)}

## Retrieval

| Modo | MRR@1 | MRR@5 | MRR@10 | Top-1 | Top-5 | Top-10 |
|------|------:|------:|-------:|------:|------:|-------:|
| Sem Cross | {retrieval_no_cross['MRR@1']:.4f} | {retrieval_no_cross['MRR@5']:.4f} | {retrieval_no_cross['MRR@10']:.4f} | {retrieval_no_cross['top1_accuracy']:.2%} | {retrieval_no_cross['top5_accuracy']:.2%} | {retrieval_no_cross['top10_accuracy']:.2%} |
| Com Cross | {retrieval_cross['MRR@1']:.4f} | {retrieval_cross['MRR@5']:.4f} | {retrieval_cross['MRR@10']:.4f} | {retrieval_cross['top1_accuracy']:.2%} | {retrieval_cross['top5_accuracy']:.2%} | {retrieval_cross['top10_accuracy']:.2%} |

## Decoder (answerKey)

| Modo | Accuracy | Parsed rate | Accuracy quando GT no prompt |
|------|---------:|------------:|------------------------------:|
| Sem Cross | {decoder_no_cross['accuracy']:.2%} | {decoder_no_cross['parsed_rate']:.2%} | {decoder_no_cross['accuracy_when_gt_context_in_prompt']:.2%} |
| Com Cross | {decoder_cross['accuracy']:.2%} | {decoder_cross['parsed_rate']:.2%} | {decoder_cross['accuracy_when_gt_context_in_prompt']:.2%} |
""",
    )

    print("=" * 70)
    print("EVAL MCQ COMPLETED")
    print("=" * 70)
    print(f"Profile: {profile_name}")
    print(f"No cross decoder accuracy: {decoder_no_cross['accuracy']:.2%}")
    print(f"With cross decoder accuracy: {decoder_cross['accuracy']:.2%}")
    print(f"Compare file: {cfg['results_compare']}")
    print(f"Metrics file: {cfg['metrics_compare']}")
    print("=" * 70)
