#!/bin/bash

BASE_FOLDER="$HOME/affectnet/data"
if [ ! -d "$BASE_FOLDER" ]; then
    echo "Erro: Diretório $BASE_FOLDER não encontrado!"
    exit 1
fi

MODELS=("VGG16" "VGG19" "ResNet18" "DenseNet" "EfficientNet" "ConvNext")

CHECKPOINTS=(
    "$HOME/checkpoints/vgg16.pth"
    "$HOME/checkpoints/vgg19.pth"
    "$HOME/checkpoints/resnet18.pth"
    "$HOME/checkpoints/densenet.pth"
    "$HOME/checkpoints/efficientnet.pth"
    "$HOME/checkpoints/convnext.pth"
)

SAMPLERS=("affectnet_weighted" "none")

EPOCHS=70
BATCH_SIZE=64
NUM_WORKERS=4
NOW=$(date +"%Y%m%d")

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    checkpoint="${CHECKPOINTS[$i]}"

    for sampler in "${SAMPLERS[@]}"; do
        echo "Executando: $model | sampler=$sampler | checkpoint=$checkpoint"

        python ./src/train.py \
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
