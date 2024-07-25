#!/bin/bash

gpus=4,5,6,7


project_name=deepseek_vl_1.3b_chat_action_dpo
model=deepseek_vl
dataset=deepseek_vl_dpo
training=dpo

# images >=6 bs only 1
# images == 4 bs can be 2 (2 will be slow)

export WANDB_API_KEY=debbae3ca343becc30f4d50fdb90cf36786b166e
export WANDB_PROJECT=${project_name}

deepspeed --include localhost:${gpus} --master_port=25678 deepseek_vl_dpo.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    training.deepspeed="scripts/deepspeed/zero3.json" \
    model.model_path="model_repo/deepseek-vl-1.3b-chat" \
    training.per_device_train_batch_size=1 \
    training.report_to="wandb" \
    training.save_strategy="steps" \
    training.save_steps=200 \
    training.num_train_epochs=5 \
    training.beta=0.01 \
