#!/bin/bash

gpus=0,1,3

project_name=deepseek_vl_1.3b_chat_vanlia_dpo
model=deepseek_vl
dataset=deepseek_vl_dpo
training=deepseek_vl_dpo

# images >=6 bs only 1
# images == 4 bs can be 2 (2 will be slow)

WANDB_PROJECT=${project_name} deepspeed --include localhost:${gpus} --master_port=25678 deepseek_vl_dpo.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    training.deepspeed="scripts/deepspeed/zero3.json" \
    model.model_path="model_repo/deepseek-vl-1.3b-chat" \
    training.per_device_train_batch_size=2 \
    dataset.file="vanlia_dpo.json" \
    training.report_to="wandb" \
    training.save_strategy="epoch" \
    training.num_train_epochs=1 \
