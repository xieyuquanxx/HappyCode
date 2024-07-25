#!/bin/bash

python dataset/split.py \
    --data_file /data/Users/xyq/developer/happy_code/data/action_dpo/v2/20240722_mc_dataset_v2_img8.json \
    --split_ratio 0.6 0.3 0.1