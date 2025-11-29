#!/bin/bash

# Verificar se o diretório existe
BASE_FOLDER="$HOME/ferplus/data"
if [ ! -d "$BASE_FOLDER" ]; then
    echo "Erro: Diretório $BASE_FOLDER não encontrado!"
    exit 1
fi

#  "VGG19" "ResNet18" "DenseNet" "EfficientNet" "ConvNext"
MODELS=("DenseNet")
#"majority" "probability" "crossentropy"
MODES=("multi_target")
# "weighted" 
SAMPLERS=("none")

EPOCHS=70
BATCH_SIZE=64
NUM_WORKERS=4

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    for sampler in "${SAMPLERS[@]}"; do
      echo "Executando: $model - $mode - $sampler"
      python ./src/train.py \
        -d "$BASE_FOLDER" \
        -m "$mode" \
        --model_name "$model" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --sampler "$sampler"
    done
  done
done