import json
import os
import glob

#------Edit here-------------
# Path to BSD dataset
dataset_path = '../../dataset/'

# BSD_1ms8ms, BSD_2ms16ms, or BSD_3ms24ms 
bsd_type = 'BSD_3ms24ms'

# BSD dataset attributes 'train', 'valid', or 'test'
phase = 'train'

# attributes to be assigned to JSON file
json_phase = 'train'
#----------------------------


dict_list = []

for seq in sorted(os.listdir(os.path.join(dataset_path, bsd_type, phase))):

    if os.path.isdir(os.path.join(dataset_path, bsd_type, phase, seq)):
        seq_dict = {}
        seq_dict['name'] = seq
        seq_dict['phase'] = phase

        sample_path_list = [
            os.path.splitext(os.path.basename(sample_path))[0]
            for sample_path in sorted(glob.glob(os.path.join(dataset_path, bsd_type, phase, seq, 'Blur', 'RGB', '*.png')))
            ]
        seq_dict['sample'] = sample_path_list

        dict_list.append(seq_dict)
    print(f'{phase=}, {seq=} appended.')


json_file = open(bsd_type + '_' + json_phase + '.json', 'w')
json.dump(dict_list, json_file, indent=4)
