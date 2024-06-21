#!/bin/bash

gpus=4,5

project_name=deepseek_vl_7b_dpo
model=deepseek_vl
dataset=deepseek_vl_dpo
training=dpo


WANDB_PROJECT=${project_name} deepspeed --include localhost:${gpus} deepseek_vl_dpo.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    training.deepspeed="scripts/deepspeed/zero3.json" \
