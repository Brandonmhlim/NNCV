#!/bin/bash
wandb login

# read model type, 
MODEL_TYPE=$(python3 -c "import config; print(config.MODEL_TYPE)")
MODEL_TYPE=$(echo "$MODEL_TYPE" | tr -cd '[:alnum:]')

if [ "$MODEL_TYPE" = "segformer" ]; then
    echo "Training SegFormer model with data augmentation..."
    python3 unified_train.py \
        --data-dir ./data/cityscapes \
        --batch-size 16 \
        --epochs 30 \
        --lr 1e-4 \
        --num-workers 12 \
        --seed 42 \
        --experiment-id "segformer-b1-a100-optimized(no augmentation)"
elif [ "$MODEL_TYPE" = "unet" ]; then
    echo "Training UNet model with data augmentation..."
    python3 unified_train.py \
        --data-dir ./data/cityscapes \
        --batch-size 16 \
        --epochs 50 \
        --lr 1e-4 \
        --num-workers 12 \
        --seed 42 \
        --experiment-id "unet(no augmentation)"
else
    echo "Unknown model type: $MODEL_TYPE"
    exit 1
fi
