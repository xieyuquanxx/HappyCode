#!/bin/bash

# gpus=4,5,6,7
gpus=0,1,2,3

project_name=deepseek_vl_1.3b_sft_lora_mc
model=deepseek_vl
dataset=sft
training=sft

export WANDB_API_KEY=debbae3ca343becc30f4d50fdb90cf36786b166e
export WANDB_PROJECT=${project_name}

deepspeed --include localhost:${gpus} --master_port=25999 deepseek_vl_sft.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    training.num_train_epochs=10 \
    training.per_device_train_batch_size=10 \
    model.lora.lora_enable=True \
    training.report_to=wandb \
    training.save_strategy="steps" \
    training.save_steps=500 \
    # run_name=2024-07-15-15-49