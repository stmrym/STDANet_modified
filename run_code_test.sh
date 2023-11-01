#!/bin/bash

python runner.py \
    --data_path=$HOME/datasets/REDS \
    --json_path=./datasets/REDS_RR_train_val_test.json \
    --data_name=REDS_RR \
    --phase=test \
    --weights=$HOME/STDAN/exp_log/train/2023-10-18T155558_STDAN_Stack_REDS/checkpoints/ckpt-epoch-0400.pth.tar  # for Huawei



# python runner.py --data_path=/mnt/d/dataset/real_RR_test --json_path=./datasets/real_RR.json --data_name=REDS_RR --phase=test --weights=$HOME/STDAN/exp_log/train/2023-10-12T174817_STDAN_Stack_REDS_large/checkpoints/ckpt-epoch-0100.pth.tar