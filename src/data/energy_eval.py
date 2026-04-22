"""Energy eval dataset parser."""

from pathlib import Path

import pandas as pd


def normalize_choices(raw_choices):
    if not isinstance(raw_choices, dict):
        return None

    labels = raw_choices.get("label", [])
    texts = raw_choices.get("text", [])
    labels = list(labels) if labels is not None else []
    texts = list(texts) if texts is not None else []

    if len(labels) != len(texts) or len(labels) == 0:
        return None

    choices = {}
    for label, text in zip(labels, texts):
        k = str(label).strip().upper()
        v = str(text).strip()
        if k and v:
            choices[k] = v
    return choices or None


def load_energy_eval_samples(path: Path, max_rows: int = 0):
    df = pd.read_parquet(path)
    if max_rows > 0:
        df = df.head(max_rows)

    samples = []
    for row in df.to_dict(orient="records"):
        question = str(row.get("question", "")).strip()
        right_context = str(row.get("right_context", "")).strip()
        answer_key = str(row.get("answerKey", "")).strip().upper()
        choices = normalize_choices(row.get("choices"))
        if not question or not right_context or not choices or answer_key not in choices:
            continue
        samples.append(
            {
                "id": str(row.get("id", "")),
                "question_number": int(row.get("question_number", -1)),
                "question": question,
                "right_context": right_context,
                "choices": choices,
                "answerKey": answer_key,
            }
        )

    return samples


def build_context_corpus(samples):
    context_to_idx = {}
    corpus = []

    for sample in samples:
        context = sample["right_context"]
        if context not in context_to_idx:
            context_to_idx[context] = len(corpus)
            corpus.append(context)

    for sample in samples:
        sample["target_context_idx"] = context_to_idx[sample["right_context"]]

    return corpus, samples
