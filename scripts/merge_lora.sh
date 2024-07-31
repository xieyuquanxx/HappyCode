#!/bin/bash


CUDA_VISIBLE_DEVICES=4 python deepseek_vl_merge_lora.py \
    --model_name model_repo/deepseek-vl-1.3b-base \
    --lora_path /data/Users/xyq/developer/happy_code/checkpoints/20240730action_mask/2024-07-30-21-13 \
    --new_model_name model_repo/20240730action_mask_9000