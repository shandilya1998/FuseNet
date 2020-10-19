#!/bin/bash

BUCKET_ID=assignment3cs6886

python train_adam.py \
    --gpu False \
    --batch-size 125 \
    --learning-rate 0.001 \
    --eps 1e-8 \
    --beta1 0.9 \
    --beta2 0.999 \
    --height 224 \
    --width 224 \
    --channels 3 \
    --job-dir gs://${BUCKET_ID}/models/gpu \
    --epochs  100 \
    --gamma 0.2 \
    --weight-decay 5e-4 \
    --seed 42 \
    --log-interval 20
