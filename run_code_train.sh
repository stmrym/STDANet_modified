#!/bin/bash

python3 runner.py \
    --data_path=$HOME/datasets/BSD_2ms16ms \
    --json_path=./datasets/BSD_2ms16ms_train_val_test.json \
    --data_name=BSD_2ms16ms \
    --phase=train   

# python3 runner.py \
#     --data_path=$HOME/datasets/BSD_3ms24ms \
#     --json_path=./datasets/BSD_3ms24ms_train_val_test.json \
#     --data_name=BSD_3ms24ms \
#     --phase=train   

python3 runner.py \
    --data_path=/work/dataset/BSD_3ms24ms \
    --json_path=/work/STDAN_modified/datasets/BSD_3ms24ms_train_val_test.json \
    --data_name=BSD_3ms24ms \
    --phase=train   



# python runner.py --data_path=$HOME/datasets/REDS --data_name=REDS_RR --phase=train
# python runner.py --data_path=$HOME/datasets/REDS --data_name=REDS_RR --phase=resume

# python runner.py --data_path=$HOME/datasets/REDS --json_path=./datasets/REDS_RR_train_val_test.json --data_name=REDS_RR --phase=train
# python runner.py --data_path=$HOME/datasets/REDS --json_path=./datasets/REDS_RR_train_val_test.json --data_name=REDS_RR --phase=resume