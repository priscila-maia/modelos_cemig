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

## Upload de modelos para Hugging Face

O encoder `qwen3_embedding_0_6b_ft_v2` fica no caminho esperado pelo profile atual:

```text
experiments/exp_v2_40k/models/qwen3_embedding_0_6b_ft_v2
```

Para subir esse modelo, ou qualquer outra pasta de modelo, para um repo unico com subpastas:

```bash
python3 scripts/upload_model_to_hf.py
python3 scripts/upload_model_to_hf.py \
  --model-path experiments/exp_v2_40k/models/qwen3_embedding_0_6b_ft_v2 \
  --repo-id CemigP/qwen3-embedding-0.6b-ft-v2 \
  --path-in-repo qwen3_embedding_0_6b_ft_v2
python3 scripts/upload_model_to_hf.py --dry-run
```

Autenticacao:

```bash
cp .env.example .env
# preencha HF_TOKEN no arquivo .env
```

Notas:

- O script cria o repo se ele ainda nao existir.
- O padrao e criar repo privado; use `--public` apenas se quiser o contrario.
- Se `--path-in-repo` nao for informado, o nome da subpasta no Hub sera o nome da pasta local do modelo.
- Para modelos `sentence-transformers`, o script valida o carregamento local antes do upload.
- Se `--token` nao for informado, o script tenta usar `HF_TOKEN` do arquivo `.env` na raiz do projeto.

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
