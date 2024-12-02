import json
import os
import glob

dataset_path = '../dataset/BSD_1ms8ms/train/blur'
# dataset_path = '../dataset/Mi11Lite/test'
phase = 'train'
save_name = './datasets/BSD_1ms8ms_train_2000'

dict_list = []


specific_seq_l = [
    # camera specific direction
    '001', '004', '008', '010', '012', '031', '035', '041', '047', '048', '050', '053', '055',
    # camera fixed
    '005', '007', '024', '030', '049', '060', '062', '089', '099', '102', '114', '117', '118',
    # camera randomly moving
    '002', '009', '013', '021', '023', '026', '038', '040', '045', '052', '061', '070', '071', '074'
    ]


seq_list = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
for seq in seq_list:

    if specific_seq_l != [] and seq not in specific_seq_l:
        continue
    else:
        print(seq)
        seq_dict = {}
        seq_dict['name'] = seq
        seq_dict['phase'] = phase

        sample_path_list = [
            os.path.splitext(os.path.basename(sample_path))[0]
            for sample_path in sorted(glob.glob(os.path.join(dataset_path, seq,'*.png')))
            ]
        
        # sample_path_list = [
        #     os.path.splitext(os.path.basename(sample_path))[0]
        #     for sample_path in sorted(glob.glob(os.path.join(dataset_path, seq, 'blur', '*.png')))
        #     ]
        
        L = len(sample_path_list)
        half_L = L//2
        start_idx = half_L - (half_L//2)
        end_idx = start_idx + half_L
        seq_dict['sample'] = sample_path_list[start_idx:end_idx]
        
        
        # seq_dict['sample'] = sample_path_list


        dict_list.append(seq_dict)


json_file = open(f'{save_name}.json', 'w')
json.dump(dict_list, json_file, indent=4)
