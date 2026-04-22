#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação do encoder Qwen FT v2 no dataset energy_eval com decoder.

Executa dois cenários:
1) Sem cross-encoder
2) Com cross-encoder (reranker_ft)

Para cada cenário, calcula:
- Métricas de retrieval (MRR@1/5/10, top-k)
- Métricas de resposta final (accuracy de answerKey via decoder)
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
TOP_K_METRICS = [1, 5, 10]

BATCH_SIZE_ENCODE = int(os.getenv("BATCH_SIZE_ENCODE", "64"))
BATCH_SIZE_RERANK = int(os.getenv("BATCH_SIZE_RERANK", "32"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "100"))
TOP_N_CONTEXTS_FOR_DECODER = int(os.getenv("TOP_N_CONTEXTS_FOR_DECODER", "3"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "2200"))
DECODER_MAX_NEW_TOKENS = int(os.getenv("DECODER_MAX_NEW_TOKENS", "8"))
EVAL_MAX_ROWS = int(os.getenv("EVAL_MAX_ROWS", "0"))

BASE_DIR = Path(__file__).parent
DEFAULT_HF_CACHE_DIR = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", str(DEFAULT_HF_CACHE_DIR)))

DATA_FILE = BASE_DIR / "datasets" / "energy_eval" / "train-00000-of-00001.parquet"
EXP_V2 = BASE_DIR / "experiments" / "exp_v2_40k"
EXP_RERANK = BASE_DIR / "experiments" / "exp_v2_40k_reranker_ft"
EXP_OUT = BASE_DIR / "experiments" / "exp_energy_eval_qwen"

MODEL_ENCODER = EXP_V2 / "models" / "qwen3_embedding_0_6b_ft_v2"
MODEL_RERANKER = Path(os.getenv("RERANKER_MODEL", str(EXP_RERANK / "models" / "reranker_ft")))
MODEL_DECODER = os.getenv("DECODER_MODEL_NAME", "Qwen/Qwen3.5-9B")

RESULTS_NO_CROSS = EXP_OUT / "results_energy_eval_qwen_no_cross.json"
RESULTS_CROSS = EXP_OUT / "results_energy_eval_qwen_cross.json"
RESULTS_COMPARE = EXP_OUT / "results_energy_eval_qwen_compare.json"
METRICS_COMPARE = EXP_OUT / "metrics_energy_eval_qwen_compare.md"

PREDICTIONS_NO_CROSS = EXP_OUT / "predictions_energy_eval_qwen_no_cross.jsonl"
PREDICTIONS_CROSS = EXP_OUT / "predictions_energy_eval_qwen_cross.jsonl"


def setup_cache_dirs():
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR / "sentence_transformers"))
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (HF_CACHE_DIR / "transformers").mkdir(parents=True, exist_ok=True)
    (HF_CACHE_DIR / "sentence_transformers").mkdir(parents=True, exist_ok=True)


def calculate_mrr(rankings, k_values=[1, 5, 10]):
    mrr_scores = {}
    for k in k_values:
        reciprocal_ranks = []
        for rank in rankings:
            reciprocal_ranks.append(1.0 / rank if rank <= k else 0.0)
        mrr_scores[f"MRR@{k}"] = np.mean(reciprocal_ranks)
    return mrr_scores


def sanity_test_mrr():
    mrr_1 = calculate_mrr([1], k_values=TOP_K_METRICS)
    if abs(mrr_1["MRR@1"] - 1.0) > 1e-9:
        raise ValueError("Sanity test falhou: rank=1")

    mrr_2 = calculate_mrr([2], k_values=TOP_K_METRICS)
    if not (abs(mrr_2["MRR@1"] - 0.0) < 1e-9 and abs(mrr_2["MRR@5"] - 0.5) < 1e-9):
        raise ValueError("Sanity test falhou: rank=2")

    mrr_3 = calculate_mrr([999], k_values=TOP_K_METRICS)
    if not (mrr_3["MRR@1"] == 0.0 and mrr_3["MRR@5"] == 0.0 and mrr_3["MRR@10"] == 0.0):
        raise ValueError("Sanity test falhou: rank grande")


def normalize_choices(raw_choices):
    if not isinstance(raw_choices, dict):
        return None

    labels = raw_choices.get("label", [])
    texts = raw_choices.get("text", [])
    labels = list(labels) if labels is not None else []
    texts = list(texts) if texts is not None else []

    if len(labels) != len(texts) or len(labels) == 0:
        return None

    out = {}
    for label, text in zip(labels, texts):
        key = str(label).strip().upper()
        value = str(text).strip()
        if key and value:
            out[key] = value

    if not out:
        return None
    return out


def load_energy_eval(path: Path, max_rows: int = 0):
    df = pd.read_parquet(path)
    if max_rows > 0:
        df = df.head(max_rows)

    rows = []
    for row in df.to_dict(orient="records"):
        question = str(row.get("question", "")).strip()
        right_context = str(row.get("right_context", "")).strip()
        answer_key = str(row.get("answerKey", "")).strip().upper()
        choices = normalize_choices(row.get("choices"))

        if not question or not right_context or not choices or answer_key not in choices:
            continue

        rows.append(
            {
                "id": str(row.get("id", "")),
                "question_number": int(row.get("question_number", -1)),
                "question": question,
                "right_context": right_context,
                "choices": choices,
                "answerKey": answer_key,
            }
        )

    return rows


def add_target_context_indices(samples):
    corpus = []
    context_to_idx = {}

    for sample in samples:
        ctx = sample["right_context"]
        if ctx not in context_to_idx:
            context_to_idx[ctx] = len(corpus)
            corpus.append(ctx)

    for sample in samples:
        sample["target_context_idx"] = context_to_idx[sample["right_context"]]

    return corpus, samples


def build_retrieval_metrics(rankings):
    mrr_scores = calculate_mrr(rankings, k_values=TOP_K_METRICS)
    return {
        **mrr_scores,
        "mean_rank": float(np.mean(rankings)),
        "median_rank": float(np.median(rankings)),
        "top1_accuracy": sum(1 for r in rankings if r == 1) / len(rankings),
        "top5_accuracy": sum(1 for r in rankings if r <= 5) / len(rankings),
        "top10_accuracy": sum(1 for r in rankings if r <= 10) / len(rankings),
    }


def evaluate_retrieval_no_cross(query_embeddings, context_embeddings, samples):
    rankings = []
    retrieved_context_indices = []

    for i in tqdm(range(len(samples)), desc="Retrieval sem cross"):
        target_idx = samples[i]["target_context_idx"]
        query_emb = query_embeddings[i]

        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
        sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        rank = np.where(sorted_indices == target_idx)[0][0] + 1

        rankings.append(int(rank))
        retrieved_context_indices.append(sorted_indices[:TOP_N_CONTEXTS_FOR_DECODER].tolist())

    return rankings, retrieved_context_indices


def evaluate_retrieval_with_cross(query_embeddings, context_embeddings, samples, questions, corpus_contexts, reranker):
    rankings = []
    retrieved_context_indices = []

    for i in tqdm(range(len(samples)), desc="Retrieval com cross"):
        target_idx = samples[i]["target_context_idx"]
        query = questions[i]
        query_emb = query_embeddings[i]

        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
        top_k = min(TOP_K_RETRIEVE, len(corpus_contexts))
        top_k_indices = torch.topk(similarities, k=top_k).indices.cpu().numpy()

        pairs = [[query, corpus_contexts[idx]] for idx in top_k_indices]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False, batch_size=BATCH_SIZE_RERANK)
        reranked_order = np.argsort(rerank_scores)[::-1]
        reranked_indices = top_k_indices[reranked_order]

        match = np.where(reranked_indices == target_idx)[0]
        rank = match[0] + 1 if len(match) > 0 else TOP_K_RETRIEVE + 1

        rankings.append(int(rank))
        retrieved_context_indices.append(reranked_indices[:TOP_N_CONTEXTS_FOR_DECODER].tolist())

    return rankings, retrieved_context_indices


def trim_context(text: str, max_chars: int):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " [...]"


def build_decoder_prompt(question: str, choices: dict, contexts: list[str]):
    contexts_md = []
    for i, ctx in enumerate(contexts, start=1):
        contexts_md.append(f"Contexto {i}:\n{trim_context(ctx, CONTEXT_MAX_CHARS)}")
    contexts_block = "\n\n".join(contexts_md)

    ordered_labels = ["A", "B", "C", "D", "E"]
    choices_block = []
    for label in ordered_labels:
        if label in choices:
            choices_block.append(f"{label}) {choices[label]}")

    return f"""Você é um especialista do setor elétrico brasileiro.

Use somente os contextos para responder a pergunta de múltipla escolha.
Se faltar informação, escolha a alternativa mais suportada pelos contextos.
Responda APENAS com uma letra única entre A, B, C, D, E.

Pergunta:
{question}

Alternativas:
{chr(10).join(choices_block)}

Contextos:
{contexts_block}

Resposta:
"""


def extract_option(text: str):
    upper = text.upper().strip()
    patterns = [
        r"RESPOSTA\s*[:\-]?\s*([ABCDE])\b",
        r"ALTERNATIVA\s*([ABCDE])\b",
        r"\b([ABCDE])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    compact = "".join(ch for ch in upper if not ch.isspace())
    if compact and compact[0] in {"A", "B", "C", "D", "E"}:
        return compact[0]
    return "N/A"


def decode_answer(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=DECODER_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    choice = extract_option(generated)
    if choice != "N/A":
        return choice, generated

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_option(full_text), generated


def evaluate_decoder(samples, corpus_contexts, retrieved_context_indices, decoder_model, decoder_tokenizer, mode_name):
    predictions = []

    for i in tqdm(range(len(samples)), desc=f"Decoder {mode_name}"):
        sample = samples[i]
        idxs = retrieved_context_indices[i]
        contexts = [corpus_contexts[idx] for idx in idxs]
        prompt = build_decoder_prompt(sample["question"], sample["choices"], contexts)

        pred_choice, raw_generation = decode_answer(decoder_model, decoder_tokenizer, prompt)
        answer_key = sample["answerKey"]
        is_correct = pred_choice == answer_key
        gt_in_prompt = sample["target_context_idx"] in idxs

        predictions.append(
            {
                "id": sample["id"],
                "question_number": sample["question_number"],
                "answer_key": answer_key,
                "predicted": pred_choice,
                "correct": bool(is_correct),
                "gt_context_in_prompt": bool(gt_in_prompt),
                "retrieved_context_indices": idxs,
                "decoder_raw": raw_generation,
            }
        )

    return predictions


def build_decoder_metrics(predictions):
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


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_decoder(model_name: str):
    decoder_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(HF_CACHE_DIR / "transformers"),
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(HF_CACHE_DIR / "transformers"),
        trust_remote_code=True,
        torch_dtype=decoder_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset de avaliação não encontrado: {DATA_FILE}")
    if not MODEL_ENCODER.exists():
        raise FileNotFoundError(f"Encoder fine-tuned não encontrado: {MODEL_ENCODER}")
    if not MODEL_RERANKER.exists():
        raise FileNotFoundError(f"Reranker fine-tuned não encontrado: {MODEL_RERANKER}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA não disponível. Esta avaliação foi configurada para usar GPU.")

    EXP_OUT.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    setup_cache_dirs()
    sanity_test_mrr()

    samples = load_energy_eval(DATA_FILE, max_rows=EVAL_MAX_ROWS)
    if not samples:
        raise RuntimeError("Nenhum exemplo válido foi carregado do energy_eval.")

    corpus_contexts, samples = add_target_context_indices(samples)
    questions = [s["question"] for s in samples]

    print(f"Amostras avaliadas: {len(samples)}")
    print(f"Contextos únicos no corpus: {len(corpus_contexts)}")
    print(f"Cache Hugging Face: {HF_CACHE_DIR}")
    print(f"Decoder: {MODEL_DECODER}")

    encoder = SentenceTransformer(
        str(MODEL_ENCODER),
        device="cuda",
        tokenizer_kwargs={"fix_mistral_regex": True},
    )

    print("Codificando corpus de contextos...")
    context_embeddings = encoder.encode(
        corpus_contexts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE_ENCODE,
    )

    print("Codificando perguntas...")
    query_embeddings = encoder.encode(
        questions,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE_ENCODE,
    )

    decoder_tokenizer, decoder_model = load_decoder(MODEL_DECODER)

    start_no_cross = time.time()
    rankings_no_cross, retrieved_no_cross = evaluate_retrieval_no_cross(
        query_embeddings,
        context_embeddings,
        samples,
    )
    predictions_no_cross = evaluate_decoder(
        samples,
        corpus_contexts,
        retrieved_no_cross,
        decoder_model,
        decoder_tokenizer,
        mode_name="sem_cross",
    )
    no_cross_time = time.time() - start_no_cross

    retrieval_no_cross = build_retrieval_metrics(rankings_no_cross)
    decoder_no_cross = build_decoder_metrics(predictions_no_cross)

    result_no_cross = {
        "mode": "no_cross_encoder",
        "dataset": str(DATA_FILE),
        "encoder_model": str(MODEL_ENCODER),
        "decoder_model": MODEL_DECODER,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "num_samples": len(samples),
        "num_unique_contexts": len(corpus_contexts),
        "top_k_metrics": TOP_K_METRICS,
        "top_n_contexts_for_decoder": TOP_N_CONTEXTS_FOR_DECODER,
        "total_time_sec": no_cross_time,
        "retrieval_metrics": retrieval_no_cross,
        "decoder_metrics": decoder_no_cross,
    }
    write_json(RESULTS_NO_CROSS, result_no_cross)
    write_jsonl(PREDICTIONS_NO_CROSS, predictions_no_cross)

    start_cross = time.time()
    reranker = CrossEncoder(str(MODEL_RERANKER), device="cuda")
    rankings_cross, retrieved_cross = evaluate_retrieval_with_cross(
        query_embeddings,
        context_embeddings,
        samples,
        questions,
        corpus_contexts,
        reranker,
    )
    predictions_cross = evaluate_decoder(
        samples,
        corpus_contexts,
        retrieved_cross,
        decoder_model,
        decoder_tokenizer,
        mode_name="com_cross",
    )
    cross_time = time.time() - start_cross

    retrieval_cross = build_retrieval_metrics(rankings_cross)
    decoder_cross = build_decoder_metrics(predictions_cross)

    result_cross = {
        "mode": "with_cross_encoder",
        "dataset": str(DATA_FILE),
        "encoder_model": str(MODEL_ENCODER),
        "reranker_model": str(MODEL_RERANKER),
        "decoder_model": MODEL_DECODER,
        "top_k_retrieve": TOP_K_RETRIEVE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "num_samples": len(samples),
        "num_unique_contexts": len(corpus_contexts),
        "top_k_metrics": TOP_K_METRICS,
        "top_n_contexts_for_decoder": TOP_N_CONTEXTS_FOR_DECODER,
        "total_time_sec": cross_time,
        "retrieval_metrics": retrieval_cross,
        "decoder_metrics": decoder_cross,
    }
    write_json(RESULTS_CROSS, result_cross)
    write_jsonl(PREDICTIONS_CROSS, predictions_cross)

    compare = {
        "dataset": str(DATA_FILE),
        "encoder_model": str(MODEL_ENCODER),
        "reranker_model": str(MODEL_RERANKER),
        "decoder_model": MODEL_DECODER,
        "cache_dir": str(HF_CACHE_DIR),
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "num_samples": len(samples),
        "num_unique_contexts": len(corpus_contexts),
        "no_cross": result_no_cross,
        "with_cross": result_cross,
    }
    write_json(RESULTS_COMPARE, compare)

    md = f"""# Energy Eval - Qwen FT + Decoder

**Dataset:** `{DATA_FILE}`  
**Encoder:** `{MODEL_ENCODER}`  
**Reranker:** `{MODEL_RERANKER}`  
**Decoder:** `{MODEL_DECODER}`  
**Cache HF:** `{HF_CACHE_DIR}`  
**Amostras:** {len(samples)}  
**Contextos únicos:** {len(corpus_contexts)}

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
"""
    with METRICS_COMPARE.open("w", encoding="utf-8") as f:
        f.write(md)

    print("=" * 70)
    print("AVALIAÇÃO ENERGY_EVAL CONCLUÍDA")
    print("=" * 70)
    print(f"Sem cross - Decoder accuracy: {decoder_no_cross['accuracy']:.2%}")
    print(f"Com cross - Decoder accuracy: {decoder_cross['accuracy']:.2%}")
    print("-" * 70)
    print(f"Resultados sem cross: {RESULTS_NO_CROSS}")
    print(f"Resultados com cross: {RESULTS_CROSS}")
    print(f"Comparação: {RESULTS_COMPARE}")
    print(f"Métricas: {METRICS_COMPARE}")
    print(f"Predições sem cross: {PREDICTIONS_NO_CROSS}")
    print(f"Predições com cross: {PREDICTIONS_CROSS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
