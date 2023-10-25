# python runner.py --data_path=$HOME/datasets/DeepVideoDeblurring_Dataset/quantitative_datasets --data_name=DVD --phase=resume
# python runner.py --data_path=$HOME/datasets/DeepVideoDeblurring_Dataset/quantitative_datasets --data_name=DVD --phase=train

# python runner.py --data_path=$HOME/datasets/REDS --data_name=REDS_RR --phase=train
# python runner.py --data_path=$HOME/datasets/REDS --data_name=REDS_RR --phase=resume

# python runner.py --data_path=$HOME/datasets/REDS --json_path=./datasets/REDS_RR_train_val_test.json --data_name=REDS_RR --phase=train
python runner.py --data_path=$HOME/datasets/REDS --json_path=./datasets/REDS_RR_train_val_test.json --data_name=REDS_RR --phase=resume