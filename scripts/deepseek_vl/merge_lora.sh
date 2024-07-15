#!/bin/bash


python deepseek_vl_merge_lora.py \
    --model_name model_repo/deepseek-vl-1.3b-base \
    --lora_path checkpoints/deepseek_vl_1.3b_sft_lora_mc/2024-07-15-15-49/checkpoint-3900 \
    --new_model_name checkpoints/deepseek_vl_1.3b_sft_mc