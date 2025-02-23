import os
import numpy as np
import glob
import argparse
import random
import json
from tqdm import tqdm
import csv
import cv2
import soundfile as sf

parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--exam', default=False, action='store_true')
parser.add_argument('--rt60', type=float, default=0.1, required=True)


random.seed(1234)


def write_csv(data_list, filepath):
    # import pdb; pdb.set_trace()
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for info in data_list:
            writer.writerow(info)
    print('{} items saved to {}.'.format(len(data_list), filepath))


def exam(args):
    read_path = f'ProcessedData/RT60_{args.rt60}'
    split_path = './data-split'
    os.makedirs(split_path, exist_ok=True)

    data_list = glob.glob(f'{read_path}/*/*.wav')
    data_list.sort()
    broken_list = []
    sample_rate_list = []
    for item in tqdm(data_list):
        meta_path = os.path.join(item, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        audio_length = meta_dict['Audio Length']
        sample_rate_list.append(meta_dict['Audio Sample Rate'])
        # import pdb; pdb.set_trace() 
        cond_1 = audio_length < 1.2
        if cond_1:
            broken_list.append(
                {
                    'broken sample': item,
                    'audio length': audio_length
                }
            )
    if len(broken_list) > 0:
        write_csv(broken_list, f'{split_path}/broken.csv')
    else:
        tqdm.write('None')
    tqdm.write(f'Sample rate: max = {np.max(sample_rate_list)}, min = {np.min(sample_rate_list)}')


def exam_TIMIT(args):
    read_path = 'TIMIT'
    split_path = './data-split'
    os.makedirs(split_path, exist_ok=True)
    data_list = glob.glob(f'{read_path}/data/*/*/*/*.wav')
    data_list.sort()
    broken_list = []
    sample_rate_list = []
    for item in tqdm(data_list):
        audio, audio_rate = sf.read(item, dtype='int16')
        sample_rate_list.append(audio_rate)
        # import pdb; pdb.set_trace() 

    tqdm.write(f'Sample rate: max = {np.max(sample_rate_list)}, min = {np.min(sample_rate_list)}')


def create_list_for_video(args, name, video_list):
    # import pdb; pdb.set_trace()
    sample_list = []

    for video in tqdm(video_list):
        meta_path = os.path.join(video, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        audio_length = meta_dict['Audio Length']
        if audio_length < 1.5:
            continue

        sample = {
            'path': os.path.join('./data/TDE-Simulation', video)
        }
        if name == 'train':
            sample_list.append(sample)
        elif name == 'val' or 'test':
            sample['start_time'] = 1
            sample_list.append(sample)

    random.shuffle(sample_list)
    return sample_list



def main(args):
    # import pdb; pdb.set_trace()
    read_path = f'ProcessedData/RT60_{args.rt60}'
    split_path = './data-split'
    os.makedirs(split_path, exist_ok=True)


    test_list = glob.glob(f'{read_path}/*')
    test_list.sort()

    # video_dict = get_download_list(args)
    sample_list = create_list_for_video(args, 'test', test_list)
    
    csv_name = f'{split_path}/test_RT60_{args.rt60}.csv'
    write_csv(sample_list, csv_name)

    

if __name__ == "__main__":
    args = parser.parse_args()
    if args.exam:
        exam(args)
    else:
        main(args)
