#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python vit_classification.py \
    training.num_train_epochs=10 \
    training.learning_rate=8e-5