import json
import os
import glob

#------Edit here-------------
# Path to GOPRO dataset
dataset_path = '../../dataset/GOPRO_Large'

# GOPRO dataset attribute 'train', or 'test'
phase = 'test'

# attributes to be assigned to JSON file
json_phase = 'valid'
#----------------------------


dict_list = []

for seq in sorted(os.listdir(os.path.join(dataset_path, phase))):

    if os.path.isdir(os.path.join(dataset_path, phase, seq)):
        seq_dict = {}
        seq_dict['name'] = seq
        seq_dict['phase'] = phase

        sample_path_list = [
            os.path.splitext(os.path.basename(sample_path))[0]
            for sample_path in sorted(glob.glob(os.path.join(dataset_path, phase, seq, 'blur', '*.png')))
            ]
        seq_dict['sample'] = sample_path_list

        dict_list.append(seq_dict)
    print(f'{phase=}, {seq=} appended.')


json_file = open('GOPRO_' + json_phase + '.json', 'w')
json.dump(dict_list, json_file, indent=4)
