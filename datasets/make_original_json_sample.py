import json
import os
import glob

#------Edit here-------------
# dataset structure: [PATH_TO_DATASET]/[PHASE]/[VIDEO_SEQS]/[FRAME]
# Path to dataset 
dataset_path = '[PATH_TO_DATASET]'

# dataset attribute 'train', 'valid', or 'test'
phase = 'test'

# saved JSON file name
savename = 'sample1'
#----------------------------

dict_list = []

seq_list = sorted([f for f in os.listdir(os.path.join(dataset_path, phase)) if os.path.isdir(os.path.join(dataset_path, phase, f))])
for seq in seq_list:
    print(seq)
    seq_dict = {}
    seq_dict['name'] = seq
    seq_dict['phase'] = phase

    sample_path_list = [
        os.path.splitext(os.path.basename(sample_path))[0]
        for sample_path in sorted(glob.glob(os.path.join(dataset_path, phase, seq,'*.png')))
        ]
    seq_dict['sample'] = sample_path_list

    dict_list.append(seq_dict)


json_file = open(f'{savename}.json', 'w')
json.dump(dict_list, json_file, indent=4)
