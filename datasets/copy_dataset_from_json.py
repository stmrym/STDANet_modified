import json
import glob
import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':

    
    config_l = [
        {
            'src_dir': '../../dataset/GOPRO_Large/train/%s/blur_gamma/%s.png',
            'dst_dir': '../../dataset/Mixed/train/blur/%s/%s.png',
            'json_path': 'GOPRO_train_1000.json' 
        },
        {
            'src_dir': '../../dataset/GOPRO_Large/train/%s/sharp/%s.png',
            'dst_dir': '../../dataset/Mixed/train/GT/%s/%s.png',
            'json_path': 'GOPRO_train_1000.json' 
        },
        {
            'src_dir': '../../dataset/BSD_3ms24ms/train/%s/Blur/RGB/%s.png',
            'dst_dir': '../../dataset/Mixed/train/blur/%s/%s.png',
            'json_path': 'BSD_3ms24ms_train_1000.json' 
        },     
        {
            'src_dir': '../../dataset/BSD_3ms24ms/train/%s/Sharp/RGB/%s.png',
            'dst_dir': '../../dataset/Mixed/train/GT/%s/%s.png',
            'json_path': 'BSD_3ms24ms_train_1000.json' 
        },        
        {
            'src_dir': '../../dataset/BSD_3ms24ms/train/%s/Sharp/RGB/%s.png',
            'dst_dir': '../../dataset/Mixed/train/GT/%s/%s.png',
            'json_path': 'BSD_3ms24ms_train_1000.json' 
        },     
        {
            'src_dir': '../../dataset/BSD_3ms24ms/train/%s/Sharp/RGB/%s.png',
            'dst_dir': '../../dataset/Mixed/train/GT/%s/%s.png',
            'json_path': 'BSD_3ms24ms_train_1000.json' 
        }
    ]

    for config in config_l:
        src_dir = config['src_dir']
        dst_dir = config['dst_dir']
        json_path = config['json_path']

    
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        for seq_dict in tqdm(json_data):
            phase = seq_dict['phase']
            seq = seq_dict['name']

            for frame in seq_dict['sample']:
                src_path = src_dir % (seq, frame)
                dst_path = dst_dir % (seq, frame)
                dst_seq_dir = os.path.dirname(dst_path)

                os.makedirs(dst_seq_dir, exist_ok=True)
                shutil.copy2(src_path, dst_path)
