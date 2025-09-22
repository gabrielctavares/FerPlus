#!/bin/bash

# Verificar se o diretório existe
BASE_FOLDER="$HOME/ferplus/data"
if [ ! -d "$BASE_FOLDER" ]; then
    echo "Erro: Diretório $BASE_FOLDER não encontrado!"
    exit 1
fi

MODELS=("VGG16" "VGG19" "ResNet18" "DenseNet" "EfficientNet" "ConvNext")
MODES=("majority" "probability" "crossentropy" "multi_target")

EPOCHS=70
BATCH_SIZE=64
NUM_WORKERS=4

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "Executando: $model - $mode"
    python ./src/train.py \
      -d "$BASE_FOLDER" \
      -m "$mode" \
      --model_name "$model" \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --num_workers $NUM_WORKERS 
  done
done