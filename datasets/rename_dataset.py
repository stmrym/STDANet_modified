import os
import glob

dataset_path = '../dataset/night_blur'

file_type_list = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

for file_type in file_type_list:
    files = sorted(glob.glob(os.path.join(dataset_path, file_type, '*.png')))
    for id, file in enumerate(files):
        new_file_name = os.path.join(dataset_path, file_type, str(id).zfill(3) + '.png')
        os.rename(file, new_file_name)
        print(f'Renamed {file} -> {new_file_name}')
