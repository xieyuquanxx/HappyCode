#!/bin/bash

gpus=0,1,2,3


CUDA_VISIBLE_DEVICES=${gpus} deepspeed deepseek_vl_sft.py \
    project=deepseek_vl_7b_sft \
    model=deepseek_vl \
    dataset=deepseek_vl_sft \
    training=deepseek_vl \
    training.num_train_epochs=10 \