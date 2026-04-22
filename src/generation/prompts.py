"""Prompt builders for generation tasks."""


def trim_context(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " [...]"


def build_mcq_prompt(question: str, choices: dict, contexts, context_max_chars: int) -> str:
    context_lines = []
    for i, ctx in enumerate(contexts, start=1):
        context_lines.append(f"Contexto {i}:\n{trim_context(ctx, context_max_chars)}")
    context_block = "\n\n".join(context_lines)

    options = []
    for label in ["A", "B", "C", "D", "E"]:
        if label in choices:
            options.append(f"{label}) {choices[label]}")
    options_block = "\n".join(options)

    return f"""Voce e um especialista do setor eletrico brasileiro.

Use somente os contextos para responder a pergunta de multipla escolha.
Se faltar informacao, escolha a alternativa mais suportada pelos contextos.
Responda APENAS com uma letra unica entre A, B, C, D, E.

Pergunta:
{question}

Alternativas:
{options_block}

Contextos:
{context_block}

Resposta:
"""
