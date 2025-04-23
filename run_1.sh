#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python train.py \
    --device "cuda" \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --lora_r 512 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --h_noiser_ratio 0 \
    --output_dir "/data/vincent/icae_prontoqa" \
    --test_size 0.1 \
    --max_steps 10000 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 1}' \
    --warmup_steps 1000 \
    --optim "adamw_torch" \
    --weight_decay 0.03 \
    --eval_strategy "no" \
    --notes ""
