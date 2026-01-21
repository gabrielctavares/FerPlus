#!/bin/bash

BASE_FOLDER="$HOME/ferplus/affectnet"
if [ ! -d "$BASE_FOLDER" ]; then
    echo "Erro: Diretório $BASE_FOLDER não encontrado!"
    exit 1
fi

MODELS=("VGG16" "VGG19" "ResNet18" "DenseNet" "EfficientNet" "ConvNext")

CHECKPOINTS=(
    "$HOME/ferplus/checkpoints/vgg16.pt"
    "$HOME/ferplus/checkpoints/vgg19.pt"
    "$HOME/ferplus/checkpoints/resnet18.pt"
    "$HOME/ferplus/checkpoints/densenet.pt"
    "$HOME/ferplus/checkpoints/efficientnet.pt"
    "$HOME/ferplus/checkpoints/convnext.pt"
)

SAMPLERS=("none" "affectnet_weighted")

EPOCHS=70
BATCH_SIZE=64
NUM_WORKERS=4
NOW=$(date +"%Y%m%d")

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    checkpoint="${CHECKPOINTS[$i]}"

    for sampler in "${SAMPLERS[@]}"; do
        echo "Executando: $model | sampler=$sampler | checkpoint=$checkpoint"

        python ./src/train_affectnet.py \
            -d "$BASE_FOLDER" \
            --checkpoint_path "$checkpoint" \
            --model_name "$model" \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --sampler "$sampler" \
            --results_file "results_${NOW}.xlsx"
    done
done
