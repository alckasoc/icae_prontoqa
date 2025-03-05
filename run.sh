rm -rf "./output"

python train.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --lora_r 1024 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --output_dir "./output" \
    --input_type "cot_only" \
    --test_size 0.1 \
    --max_steps 1000 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 1}' \
    --warmup_steps 250 \
    --optim "adamw_torch" \
    --weight_decay 0.01 \
    --notes ""
