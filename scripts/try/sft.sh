#!/bin/bash

gpus=4,5,6,7
# gpus=0,1,2,3

project_name=20240731PredSingleAction
model=try
dataset=deepseek_vl_sft
training=sft

export WANDB_API_KEY=debbae3ca343becc30f4d50fdb90cf36786b166e
export WANDB_PROJECT=${project_name}

deepspeed --include localhost:${gpus} --master_port=25999 train/try/try_pred_single_action.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    model.attn_implementation="flash_attention_2" \
    training.num_train_epochs=30 \
    training.per_device_train_batch_size=8 \
    training.report_to=wandb \
    training.save_strategy="epoch" \
    training.learning_rate=2e-5 \
    training.warmup_ratio=0.0 \
    training.eval_strategy="epoch" \