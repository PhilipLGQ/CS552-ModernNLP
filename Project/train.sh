#!/bin/bash

python ./src/finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path 'dataset/gen_dataset_ikun.json' \
    --output_dir 'model/lora-alpaca-new' \
    --custom_dataset \
    --batch_size 128 \
    --micro_batch_size 8 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --cutoff_len 1024 \
    --val_set_size 350 \
    --lora_r 8 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --group_by_length \
    --add_eos_token
