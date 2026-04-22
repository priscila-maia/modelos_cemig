# Pipeline CEMIG

Fluxo oficial atual para treino/avaliacao continua valido para E5 + reranker.

Este repo agora tambem possui um fluxo modular em `src/`, preparado para etapas
futuras que podem ou nao ser Qwen.

## Requisitos

- Python 3.9+
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Estrutura nova (modular)

```text
src/
  core/         # config, cache, seed, metricas, IO
  data/         # loaders JSONL/parquet e parsers de dataset
  retrieval/    # encoder retrieval e rerank cross-encoder
  generation/   # decoder causal + prompts
  pipelines/    # pipelines genericos + profiles
scripts/        # CLIs para rodar pipelines
```

## Profiles (extensivel)

Os pipelines usam profile para defaults de modelo, caminhos e parametros.

- profile atual: `qwen_v2`
- para novos modelos/etapas: adicionar profile em `src/pipelines/profiles/`

## Dados esperados em `datasets/`

- `snapshot_v1.jsonl`
- `snapshot_v1_enriquecido.jsonl`
- `snapshot_v1_filtrado.jsonl`
- `train_pos_v1.jsonl`
- `train_pos_v2.jsonl`
- `train_cross_encoder.jsonl`
- `energy_eval/train-00000-of-00001.parquet`

## Cache de modelos (Hugging Face)

Por padrao, os scripts usam o caminho atual do Hugging Face (`HF_HOME`,
normalmente `~/.cache/huggingface`).

Para forcar cache local no repo:

```bash
export HF_CACHE_DIR="$(pwd)/.cache/huggingface"
```

## Fluxo oficial legado (E5 + reranker)

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

## Fluxo modular (novo)

```bash
python3 scripts/run_train_encoder.py --profile qwen_v2
python3 scripts/run_eval_retrieval.py --profile qwen_v2
python3 scripts/run_eval_mcq.py --profile qwen_v2
```

## Compatibilidade com comandos antigos (Qwen)

Os scripts antigos foram mantidos como wrappers para o fluxo modular:

```bash
python3 12_train_qwen_encoder_v2.py
python3 19_eval_qwen_ft_v2.py
python3 20_eval_qwen_energy_eval_decoder.py
```

## Saidas principais do fluxo Qwen

- `experiments/exp_v2_40k/results_qwen_ft_v2_no_cross.json`
- `experiments/exp_v2_40k/results_qwen_ft_v2_cross.json`
- `experiments/exp_v2_40k/results_qwen_ft_v2_compare.json`
- `experiments/exp_energy_eval_qwen/results_energy_eval_qwen_compare.json`
- `experiments/exp_energy_eval_qwen/metrics_energy_eval_qwen_compare.md`

## Docker (minimo)

Imagem base usada:

- `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`

Build:

```bash
docker compose build
```

Executar shell no container:

```bash
docker compose run --rm qwen-flow
```

Executar pipeline no container:

```bash
docker compose run --rm qwen-flow python3 scripts/run_eval_retrieval.py --profile qwen_v2
```
