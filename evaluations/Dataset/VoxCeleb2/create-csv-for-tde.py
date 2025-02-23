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
parser.add_argument('--max_sample', default=-1, type=int)
parser.add_argument('--type', default='', type=str)
parser.add_argument('--data_split', default='7:1:2', type=str)
parser.add_argument('--unshuffle', default=False, action='store_true')
parser.add_argument('--sim_setup', default='', type=str)

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




def create_list_for_video(args, name, video_list):
    # import pdb; pdb.set_trace()
    sample_list = []

    for video in tqdm(video_list):
        path = os.path.join('./data/VoxCeleb2', video)
        meta_path = os.path.join(video, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        frame_rate = meta_dict['frame_rate']
        frame_num_3 = np.floor(meta_dict['Audio Length'] * frame_rate).astype(int)
        speaker1_save_facepath = os.path.join(video, 'speaker1_faces')
        speaker2_save_facepath = os.path.join(video, 'speaker2_faces')
        frame_num_1 = len(glob.glob(f'{speaker1_save_facepath}/*'))
        frame_num_2 = len(glob.glob(f'{speaker2_save_facepath}/*'))
        frame_num = min(frame_num_1, frame_num_2, frame_num_3)


        if name == 'train':
            sample_list.append({'path': path})
        elif name in ['val', 'test']:
            if frame_num < frame_rate * 3:
                continue
            start_times = np.random.choice(frame_num - frame_rate * 3, 1, replace=False)
            for i in start_times:
                sample = {
                    'path': path,
                    'start_time': i
                }
                sample_list.append(sample)
    if not args.unshuffle:
        random.shuffle(sample_list)
    return sample_list


def main(args):
    # import pdb; pdb.set_trace()
    read_path = f'ProcessedData-TDE/{args.sim_setup}'
    split_path = f'./data-split'
    if args.type != '':
        split_path = os.path.join(split_path, args.type)
    
    if args.sim_setup != '':
        split_path = os.path.join(split_path, args.sim_setup)

    os.makedirs(split_path, exist_ok=True)

    data_list = glob.glob(f'{read_path}/*/*')
    data_list.sort()
    if not args.unshuffle:
        random.shuffle(data_list)
    if args.max_sample != -1:
        data_list = data_list[:args.max_sample]
    
    begin = 0
    ratios = args.data_split.split(':')
    ratios = np.array(list(map(int, ratios)))
    ratios = ratios / ratios.sum()
    n_train = begin + ratios[0]
    n_val = n_train + ratios[1]
    n_test = n_val + ratios[2]

    train_list = data_list[int(len(data_list) * begin) : int(len(data_list) * n_train)]
    valid_list = data_list[int(len(data_list) * n_train) : int(len(data_list) * n_val)]
    test_list = data_list[int(len(data_list) * n_val) : int(len(data_list) * n_test)]

    csv_zip = zip(['train', 'val', 'test'], [train_list, valid_list, test_list])
    for name, video_list in tqdm(csv_zip):
        if len(video_list) == 0:
            continue

        sample_list = create_list_for_video(args, name, video_list)
        csv_name = f'{split_path}/{name}.csv'
        write_csv(sample_list, csv_name)

    
# python create-csv-for-tde.py --type='voxceleb-tde-longer' --data_split='0:0:1' 
# python create-csv-for-tde.py --type='voxceleb-tde' --data_split='0:0:1'  --sim_setup='Easy'
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)