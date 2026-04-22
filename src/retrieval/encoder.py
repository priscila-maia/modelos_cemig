"""SentenceTransformer encoder helpers."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_sentence_encoder(model_name_or_path: str, device: str, trust_remote_code: bool = False, model_kwargs=None, tokenizer_kwargs=None):
    model_kwargs = model_kwargs or {}
    tokenizer_kwargs = tokenizer_kwargs or {}
    return SentenceTransformer(
        model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )


def encode_texts(model: SentenceTransformer, texts, batch_size: int):
    return model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )


def rank_indices_by_cosine(query_emb, context_embeddings):
    similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
    sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
    return sorted_indices


def topk_indices_by_cosine(query_emb, context_embeddings, k: int):
    similarities = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)
    return torch.topk(similarities, k=k).indices.cpu().numpy()


def rank_of_target(sorted_indices: np.ndarray, target_idx: int) -> int:
    return int(np.where(sorted_indices == target_idx)[0][0] + 1)
