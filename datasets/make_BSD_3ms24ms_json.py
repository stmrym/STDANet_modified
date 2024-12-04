import json
import os
import glob

dataset_path = '../dataset/BSD_3ms24ms/train/blur'
# dataset_path = '../dataset/Mi11Lite/test'
phase = 'train'
save_name = './datasets/BSD_3ms24ms_train_2000'

dict_list = []


specific_seq_l = [
    # camera specific direction
    '001', '009', '018', '024', '030', '033', '036', '045', '054', '063', '066', '069', '070', '079', '093', '096',
    # camera fixed
    '019', '020', '060', '068', '073', '078', '102', '105',
    # camera randomly moving
    '017', '021', '023', '031', '034', '035', '048', '051', '062', '067', '071', '076', '081', '083', '085', '086'
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
