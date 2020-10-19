#!/bin/bash

BUCKET_ID=assignment3cs6886

python train.py \
    --gpu False \
    --batch-size 64 \
    --learning-rate 0.01 \
    --momentum 0.6 \
    --height 224 \
    --width 224 \
    --channels 3 \
    --job-dir gs://${BUCKET_ID}/models/gpu \
    --epochs  100 \
    --gamma 0.2 \
    --weight-decay 5e-4 \
    --seed 42 \
    --log-interval 20 
    #--checkpoint-path gs://assignment3cs6886/models_2/gpu/model_20201018_162211/checkpoint/fuse/Sunday_18_October_2020_14h_46m_07s/fuse-190-best.pth
