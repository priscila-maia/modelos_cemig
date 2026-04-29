# Pipeline CEMIG

Este repositorio usa um fluxo modular em `src/` com CLIs em `scripts/`.

## Requisitos

- Python 3.9+
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Estrutura

```text
src/
  core/         # config, cache, seed, metricas, IO
  data/         # loaders JSONL/parquet e parsers de dataset
  retrieval/    # encoder retrieval e rerank cross-encoder
  generation/   # decoder causal + prompts
  pipelines/    # pipelines genericos + profiles
scripts/        # CLIs para rodar pipelines
configs/        # YAMLs de execucao
```

## Profiles

Os pipelines usam profile para defaults de modelo, caminhos e parametros.

- profile atual: `qwen_v2`
- para novos modelos/etapas: adicionar profile em `src/pipelines/profiles/`

## Dados esperados em `datasets/`

- `train_pos_v2.jsonl`
- `energy_eval/train-00000-of-00001.parquet`

## Cache de modelos (Hugging Face)

Por padrao, os scripts usam o caminho atual do Hugging Face (`HF_HOME`, normalmente `~/.cache/huggingface`).

Para forcar cache local no repo:

```bash
export HF_CACHE_DIR="$(pwd)/.cache/huggingface"
```

## Fluxo modular

```bash
python3 scripts/run_train_encoder.py --profile qwen_v2
python3 scripts/run_eval_retrieval.py --profile qwen_v2
python3 scripts/run_eval_mcq.py --profile qwen_v2
```

## Configuracoes YAML para `eval_mcq`

Configs prontas em `configs/eval_mcq/`:

- `qwen3_5_9b.yaml`
- `qwen3_5_4b.yaml`
- `cemig_qwen3_4b_dw_lr.yaml`

Rodar local com YAML:

```bash
python3 scripts/run_eval_mcq_from_yaml.py --config configs/eval_mcq/qwen3_5_4b.yaml
python3 scripts/run_eval_mcq_from_yaml.py --config configs/eval_mcq/cemig_qwen3_4b_dw_lr.yaml
```

## Saidas principais do fluxo Qwen

- `experiments/exp_v2_40k/results_qwen_ft_v2_no_cross.json`
- `experiments/exp_v2_40k/results_qwen_ft_v2_cross.json`
- `experiments/exp_v2_40k/results_qwen_ft_v2_compare.json`
- `experiments/exp_energy_eval_qwen/results_energy_eval_qwen_compare.json`
- `experiments/exp_energy_eval_qwen/metrics_energy_eval_qwen_compare.md`

## Docker

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

Executar `eval_mcq` com YAML no container:

```bash
docker compose run --rm qwen-flow python3 scripts/run_eval_mcq_from_yaml.py --config configs/eval_mcq/qwen3_5_4b.yaml
docker compose run --rm qwen-flow python3 scripts/run_eval_mcq_from_yaml.py --config configs/eval_mcq/cemig_qwen3_4b_dw_lr.yaml
```
