#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python deepseek_vl_merge_lora.py \
    --model_name checkpoints/memory_bank_1.3b_based_on_sft/2024-07-18-22-20 \
    --lora_path checkpoints/memory_bank_1.3b_dpo/2024-07-19-14-23/checkpoint-1400 \
    --new_model_name model_repo/memory_bank_1.3b_action_dpo