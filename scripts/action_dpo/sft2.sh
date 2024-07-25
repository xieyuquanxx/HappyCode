#!/bin/bash

gpus=4,5,6,7
# gpus=0,1,2,3

project_name=action_vlm
model=action_vlm
dataset=action_vlm_sft
training=deepseek_vl_sft

export WANDB_API_KEY=debbae3ca343becc30f4d50fdb90cf36786b166e
export WANDB_PROJECT=${project_name}

deepspeed --include localhost:${gpus} --master_port=25999 action_sft_stage2.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    dataset.file=20240722_mc_dataset_v2_img8_56345.json \
    training.num_train_epochs=10 \
    training.per_device_train_batch_size=16 \
    model.lora.lora_enable=False \
    training.report_to=wandb \
    training.save_strategy="epoch" \
    training.save_steps=500 \
    training.learning_rate=4e-6
