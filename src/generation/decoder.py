"""Causal decoder loading and multiple-choice decoding helpers."""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_decoder(model_name: str, cache_dir: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def extract_choice(text: str) -> str:
    upper = text.upper().strip()
    for pattern in [r"RESPOSTA\s*[:\-]?\s*([ABCDE])\b", r"ALTERNATIVA\s*([ABCDE])\b", r"\b([ABCDE])\b"]:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    compact = "".join(ch for ch in upper if not ch.isspace())
    if compact and compact[0] in {"A", "B", "C", "D", "E"}:
        return compact[0]
    return "N/A"


def decode_choice(prompt: str, tokenizer, model, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    choice = extract_choice(generated)
    if choice != "N/A":
        return choice, generated

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_choice(full_text), generated
