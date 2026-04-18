#!/bin/bash
# Script auxiliar para treinar o encoder oficial v2 dentro do container Docker
# Uso: bash run_train_in_docker.sh

set -e

SRC_DIR="/raid/user_priscilaribeiro/filtragem_cemig"
MOUNT_DIR="/raid/user_priscilaribeiro/Cemig"
EXP_REL="experiments/exp_v2_40k"

echo "Preparando ambiente de treino v2 no container..."

# O split v2 precisa existir antes do treino.
if [[ ! -f "$SRC_DIR/$EXP_REL/train_pos_v2_train.jsonl" ]]; then
    echo "Arquivo nao encontrado: $SRC_DIR/$EXP_REL/train_pos_v2_train.jsonl"
    echo "Execute antes: python3 11_split_v2.py"
    exit 1
fi

echo "Copiando arquivos para area montada..."
mkdir -p "$MOUNT_DIR/$EXP_REL"
cp "$SRC_DIR/12_train_e5_encoder_v2.py" "$MOUNT_DIR/"
cp "$SRC_DIR/$EXP_REL/train_pos_v2_train.jsonl" "$MOUNT_DIR/$EXP_REL/"

echo "Iniciando treino no container..."
docker exec cemig-app-chunks bash -c "
    cd /app && \
    pip install -q sentence-transformers && \
    python 12_train_e5_encoder_v2.py
"

echo "Copiando resultados de volta..."
mkdir -p "$SRC_DIR/$EXP_REL/models"
cp -r "$MOUNT_DIR/$EXP_REL/models/e5_large_ft_v2" "$SRC_DIR/$EXP_REL/models/"
cp "$MOUNT_DIR/$EXP_REL/train_config_v2.json" "$SRC_DIR/$EXP_REL/" || true
cp "$MOUNT_DIR/$EXP_REL/train_summary.json" "$SRC_DIR/$EXP_REL/" || true

echo "Treino v2 concluido. Resultados em $SRC_DIR/$EXP_REL/models/e5_large_ft_v2"
