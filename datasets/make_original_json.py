import json
import os
import glob

dataset_path = '../../dataset/night_blur/Long'
phase = 'test'

dict_list = []
seq_dict = {}
seq_dict['name'] = 'night_blur'
seq_dict['phase'] = phase

sample_path_list = [
    os.path.splitext(os.path.basename(sample_path))[0]
    for sample_path in sorted(glob.glob(os.path.join(dataset_path, '*.png')))
    ]
seq_dict['sample'] = sample_path_list

dict_list.append(seq_dict)


json_file = open('night_blur.json', 'w')
json.dump(dict_list, json_file, indent=4)
