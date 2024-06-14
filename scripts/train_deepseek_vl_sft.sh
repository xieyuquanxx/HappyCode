#!/bin/bash

gpus=3

project_name=deepseek_vl_7b_sft_lora
model=deepseek_vl
dataset=deepseek_vl_sft
training=deepseek_vl


WANDB_PROJECT=${project_name} CUDA_VISIBLE_DEVICES=${gpus} deepspeed deepseek_vl_sft.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    training.num_train_epochs=1 \
    training.per_device_train_batch_size=8 \
    model.lora.lora_enable=True \
    training.report_to=wandb \