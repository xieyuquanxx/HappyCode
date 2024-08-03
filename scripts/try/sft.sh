#!/bin/bash

gpus=0,1
# gpus=0,1,2,3

project_name=20240802PredSingleAction
model=try
dataset=deepseek_vl_sft
training=sft

export WANDB_API_KEY=debbae3ca343becc30f4d50fdb90cf36786b166e
export WANDB_PROJECT=${project_name}

deepspeed --include localhost:${gpus} --master_port=25999 try_pred_single_action.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    training.num_train_epochs=5 \
    training.per_device_train_batch_size=12 \
    training.report_to=wandb \
    training.save_strategy="epoch"