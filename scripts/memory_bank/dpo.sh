#!/bin/bash

gpus=0,1,2,3


project_name=memory_bank_1.3b_dpo
model=ds_vl_memory_bank
dataset=memory_bank_dpo
training=deepseek_vl_dpo

export WANDB_API_KEY=debbae3ca343becc30f4d50fdb90cf36786b166e
export WANDB_PROJECT=${project_name}

deepspeed --include localhost:${gpus} --master_port=25678 memory_bank_dpo.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    training.deepspeed="scripts/deepspeed/zero3.json" \
    model.model_path="checkpoints/memory_bank_1.3b_based_on_sft/2024-07-18-22-20" \
    training.per_device_train_batch_size=4 \
    training.report_to="wandb" \
    training.save_strategy="steps" \
    training.save_steps=200 \
    training.num_train_epochs=5 \
    training.beta=0.01 \
