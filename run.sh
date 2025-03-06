#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python train.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --lora_r 512 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir "/data/vincent/icae_prontoqa" \
    --input_type "cot_only" \
    --test_size 0.1 \
    --max_steps 25000 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 1}' \
    --warmup_steps 1000 \
    --optim "adamw_torch" \
    --weight_decay 0.03 \
    --eval_strategy "no" \
    --notes ""

python train.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --lora_r 512 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir "/data/vincent/icae_prontoqa" \
    --input_type "full_format" \
    --test_size 0.1 \
    --max_steps 25000 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 1}' \
    --warmup_steps 1000 \
    --optim "adamw_torch" \
    --weight_decay 0.03 \
    --eval_strategy "no" \
    --notes ""

python train.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --lora_r 512 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir "/data/vincent/icae_prontoqa" \
    --input_type "q_q_only" \
    --test_size 0.1 \
    --max_steps 25000 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 1}' \
    --warmup_steps 1000 \
    --optim "adamw_torch" \
    --weight_decay 0.03 \
    --eval_strategy "no" \
    --notes ""

