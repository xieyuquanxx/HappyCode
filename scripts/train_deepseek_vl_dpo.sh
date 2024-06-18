#!/bin/bash

gpus=1

project_name=deepseek_vl_7b_dpo
model=deepseek_vl
dataset=deepseek_vl_dpo
training=deepseek_vl_dpo


WANDB_PROJECT=${project_name} CUDA_VISIBLE_DEVICES=${gpus} python deepseek_vl_dpo.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training}