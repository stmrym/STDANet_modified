import json
import os
import glob

dataset_path = '../../dataset/ADAS/test/input'
phase = 'test'
save_name = 'ADAS'

dict_list = []

seq_list = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
for seq in seq_list:
    print(seq)
    seq_dict = {}
    seq_dict['name'] = seq
    seq_dict['phase'] = phase

    sample_path_list = [
        os.path.splitext(os.path.basename(sample_path))[0]
        for sample_path in sorted(glob.glob(os.path.join(dataset_path, seq,'*.jpg')))
        ]
    seq_dict['sample'] = sample_path_list

    dict_list.append(seq_dict)


json_file = open(f'{save_name}.json', 'w')
json.dump(dict_list, json_file, indent=4)
