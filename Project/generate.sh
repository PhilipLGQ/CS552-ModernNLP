#!/bin/bash

python ./src/generate.py \
    --load_8bit \
    --base_model 'models/lora-alpaca-adapter_merged' \
    --lora_weights 'models/ppo-fine-tuned-ethical-frugal' \
    --max_generate_length 128 \
    --prompt_dataset_dir 'dataset/prompts_processed.json' \
    --prompt_save_dir 'dataset' \
    --prompt_save_name 'answers_ikun' \
    --convert_to_submission