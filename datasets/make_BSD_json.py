import json
import os
import glob

#------Edit here-------------
# Path to BSD dataset
dataset_path = '../../dataset/'

# BSD_1ms8ms, BSD_2ms16ms, or BSD_3ms24ms 
bsd_type = 'BSD_3ms24ms'

# BSD dataset attributes 'train', 'valid', or 'test'
phase = 'valid'

# attributes to be assigned to JSON file
json_phase = 'valid_1000'
#----------------------------


dict_list = []
count = 0
for seq in sorted(os.listdir(os.path.join(dataset_path, bsd_type, phase))):

    if os.path.isdir(os.path.join(dataset_path, bsd_type, phase, seq)):
        seq_dict = {}
        seq_dict['name'] = seq
        seq_dict['phase'] = phase

        sample_path_list = [
            os.path.splitext(os.path.basename(sample_path))[0]
            for sample_path in sorted(glob.glob(os.path.join(dataset_path, bsd_type, phase, seq, 'Blur', 'RGB', '*.png')))
            ]
        sample_path_list = sample_path_list[0:25]
        seq_dict['sample'] = sample_path_list

        dict_list.append(seq_dict)
    print(f'{phase=}, {seq=} num={len(sample_path_list)} appended.')
    count += len(sample_path_list)
    if count == 1000:
        break

print(f'Total {count=}')
json_file = open(bsd_type + '_' + json_phase + '.json', 'w')
json.dump(dict_list, json_file, indent=4)
