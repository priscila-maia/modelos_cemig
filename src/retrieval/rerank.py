"""Cross-encoder reranking helpers."""

import numpy as np
from sentence_transformers import CrossEncoder


def load_cross_encoder(model_name_or_path: str, device: str):
    return CrossEncoder(model_name_or_path, device=device)


def rerank_indices(query: str, contexts, candidate_indices, reranker: CrossEncoder, batch_size: int):
    pairs = [[query, contexts[idx]] for idx in candidate_indices]
    scores = reranker.predict(pairs, show_progress_bar=False, batch_size=batch_size)
    order = np.argsort(scores)[::-1]
    return candidate_indices[order]
