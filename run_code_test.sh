#!/bin/bash

python3 runner.py \
    --data_path=/work/dataset/BSD_3ms24ms \
    --json_path=./datasets/BSD_3ms24ms_train_val_test.json \
    --data_name=BSD_3ms24ms \
    --phase=test \
    --weights=./exp_log/train/2023-11-02T095305_STDAN_Stack_BSD_3ms24ms/checkpoints/best-ckpt.pth.tar  

# python runner.py \
#     --data_path=$HOME/datasets/BSD_3ms24ms \
#     --json_path=./datasets/BSD_3ms24ms_train_val_test.json \
#     --data_name=BSD_3ms24ms \
#     --phase=test \
#     --weights=$HOME/STDAN/exp_log/train/2023-11-01T103816_STDAN_Stack_BSD_3ms24ms/checkpoints/best-ckpt.pth.tar  


# python runner.py --data_path=/mnt/d/dataset/real_RR_test --json_path=./datasets/real_RR.json --data_name=REDS_RR --phase=test --weights=$HOME/STDAN/exp_log/train/2023-10-12T174817_STDAN_Stack_REDS_large/checkpoints/ckpt-epoch-0100.pth.tar