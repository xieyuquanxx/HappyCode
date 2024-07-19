#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python deepseek_vl_merge_lora.py \
    --model_name model_repo/deepseek-vl-1.3b-base \
    --lora_path checkpoints/deepseek_vl_1.3b_sft_lora_mc/2024-07-15-15-49/checkpoint-72500 \
    --new_model_name model_repo/deepseek_vl_1.3b_sft_mc