import json
import os
import glob

dataset_path = '../dataset/BSD_2ms16ms/train/blur'
# dataset_path = '../dataset/Mi11Lite/test'
phase = 'train'
save_name = './datasets/BSD_2ms16ms_train_2000'

dict_list = []


specific_seq_l = [
    # camera specific direction
    '000', '004', '013', '024', '027', '032', '037', '039', '055', '056', '065', '070', '074',
    # camera fixed
    '007', '014', '017', '019', '023', '025', '026', '033', '040', '045', '049', '076', '079', '082',
    # camera randomly moving
    '003', '005', '008', '009', '016', '054', '057', '058', '059', '067', '071', '081', '084'
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
