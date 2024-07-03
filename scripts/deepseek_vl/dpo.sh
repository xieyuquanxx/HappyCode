#!/bin/bash

gpus=4,5,6,7

project_name=deepseek_vl_1.3b_mdpo_test_four_image
model=deepseek_vl
dataset=deepseek_vl_dpo
training=deepseek_vl_dpo

# images >=6 bs only 1
# images == 4 bs can be 2 (2 will be slow)

WANDB_PROJECT=${project_name} deepspeed --include localhost:${gpus} deepseek_vl_dpo.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    training.deepspeed="scripts/deepspeed/zero3.json" \
    model.model_path="model_repo/deepseek-vl-1.3b-base" \
    training.per_device_train_batch_size=1 \
