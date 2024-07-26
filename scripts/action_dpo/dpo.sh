#!/bin/bash

gpus=0,1,2,3


project_name=action_vlm
model=action_vlm
dataset=action_vlm_dpo
training=dpo

export WANDB_API_KEY=debbae3ca343becc30f4d50fdb90cf36786b166e
export WANDB_PROJECT=${project_name}

deepspeed --include localhost:${gpus} --master_port=25678 action_dpo.py \
    project=${project_name} \
    model=${model} \
    dataset=${dataset} \
    training=${training} \
    model.lora.lora_enable=True \
    model.model_path="/data/Users/xyq/developer/happy_code/checkpoints/action_vlm/2024-07-24-22-42" \
    training.deepspeed="scripts/deepspeed/zero3.json" \
    training.per_device_train_batch_size=4 \
    training.report_to="wandb" \
    training.save_strategy="epoch" \
    training.save_steps=200 \
    training.num_train_epochs=5 \
    training.beta=0.01 \
    training.learning_rate=5.0e-6 \
