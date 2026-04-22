"""Generic JSONL loaders used by training and retrieval eval pipelines."""

from pathlib import Path

from src.core.io import read_jsonl


def load_query_context_pairs(path: Path):
    pairs = []
    for row in read_jsonl(path):
        query = str(row.get("query", "")).strip()
        context = str(row.get("context", "")).strip()
        if query and context:
            pairs.append({"query": query, "context": context})
    return pairs
