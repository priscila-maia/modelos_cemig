# Pipeline CEMIG

Fluxo oficial para treinar e avaliar:
- Encoder: `intfloat/multilingual-e5-large` (fine-tuned v2)
- Reranker: `BAAI/bge-reranker-v2-m3` (fine-tuned)

## Requisitos

- Python 3.9+
- Instalar dependências:

```bash
pip install -r requirements.txt
```

## Dados esperados em `datasets/`

- `snapshot_v1.jsonl`
- `snapshot_v1_enriquecido.jsonl`
- `snapshot_v1_filtrado.jsonl`
- `train_pos_v1.jsonl`
- `train_pos_v2.jsonl`
- `train_cross_encoder.jsonl`
- `energy_eval/train-00000-of-00001.parquet`

## Cache de modelos (Hugging Face)

Os scripts de avaliação do Qwen agora seguem o caminho padrão atual do Hugging Face
(normalmente `~/.cache/huggingface`, ou o que já estiver em `HF_HOME`).

Se quiser forçar cache local dentro do projeto:

```bash
export HF_CACHE_DIR="$(pwd)/.cache/huggingface"
```

## Execução (ordem oficial)

```bash
python3 01_validate_snapshot_v1.py
python3 02_enrich_snapshot_v1.py
python3 03_filter_snapshot_v1.py
python3 04_build_train_pairs_v1.py
python3 05_split_dataset_v1.py
python3 06_eval_baseline_e5_v1.py
python3 07_train_e5_encoder_v1.py
python3 08_eval_e5_ft_v1.py
python3 09_rerank_bge_m3_v1.py
python3 10_filter_top25_v1.py
python3 11_split_v2.py
python3 12_train_e5_encoder_v2.py
python3 13_eval_e5_ft_v2.py
python3 14_rerank_bge_m3_v2.py
python3 15_results_summary_cycle2.py
python3 16_build_cross_encoder_dataset_v2.py
python3 17_train_reranker_ft_v2.py
python3 18_eval_reranker_ft_v2.py
```

## Saídas principais

- `experiments/exp_v2_40k/models/e5_large_ft_v2`
- `experiments/exp_v2_40k_reranker_ft/models/reranker_ft`
- `experiments/exp_v2_40k/results_summary_cycle2.md`
- `experiments/exp_v2_40k_reranker_ft/results_summary_cycle3.md`

## Opcional (experimento Qwen)

```bash
python3 12_train_qwen_encoder_v2.py
python3 19_eval_qwen_ft_v2.py
python3 20_eval_qwen_energy_eval_decoder.py
```

### O que cada script opcional gera

- `19_eval_qwen_ft_v2.py`
  - avaliação sem cross-encoder e com cross-encoder (usando `reranker_ft`)
  - arquivos em `experiments/exp_v2_40k/` com sufixos `no_cross`, `cross` e `compare`
- `20_eval_qwen_energy_eval_decoder.py`
  - avaliação do `qwen3_embedding_0_6b_ft_v2` no `energy_eval`
  - usa decoder `Qwen/Qwen3.5-9B` para responder as alternativas (`answerKey`)
  - compara sem cross-encoder vs com cross-encoder
  - saídas em `experiments/exp_energy_eval_qwen/`
