import json
import os
import glob

dataset_path = os.path.join(os.environ['HOME'], 'datasets')
bsd_type = 'BSD_3ms24ms'
phase_list = ['test', 'valid', 'train']

dict_list = []

for phase in phase_list:
    for seq in sorted(os.listdir(os.path.join(dataset_path, bsd_type, phase))):
        print(phase, seq)

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


json_file = open(bsd_type + '_train_val_test.json', 'w')
json.dump(dict_list, json_file, indent=4)
