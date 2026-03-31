wandb login

python3 train_data_augmentation_segformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.0005 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "segformer-b1-a100-optimized"