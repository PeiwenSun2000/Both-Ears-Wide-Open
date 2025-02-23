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

# python create-csv.py --dataset_name='Youtube-RacingCar' --type='visualization' --data_split='1:0:0' 
# python create-csv.py --dataset_name='Youtube-Inthewild' --type='' --data_split='1:0:0' 


parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--exam', default=False, action='store_true')
parser.add_argument('--type', default='', type=str)
parser.add_argument('--dataset_name', default='', type=str)
parser.add_argument('--data_split', default='8:1:1', type=str)
parser.add_argument('--split_by_video', default=False, action='store_true')
parser.add_argument('--unshuffle', default=False, action='store_true')

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
    # read_path = 'ProcessedData'
    read_path = args.data_path
    split_path = f'./data-split'
    os.makedirs(split_path, exist_ok=True)
    
    data_list = glob.glob(f'{read_path}/*')
    data_list.sort()
    broken_list = []
    sample_rate_list = []
    for item in tqdm(data_list):
        meta_path = os.path.join(item, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        sample_rate_list.append(meta_dict['audio_sample_rate'])
        # import pdb; pdb.set_trace() 
        audio_path = os.path.join(item, 'audio', 'audio.wav')
        audio, audio_rate = sf.read(audio_path, dtype='int16')
        cond_1 = meta_dict['frame_num'] <= 98
        cond_2 = not meta_dict['ifstereo']
        cond_3 = np.abs(audio.shape[0] / audio_rate - meta_dict['frame_num'] / meta_dict['frame_rate'] ) > 0.1
        if cond_1 or cond_2 or cond_3:
            broken_list.append(
                {
                    'broken url': meta_dict['u_id'],
                    'broken time': item,
                    'frame num': meta_dict['frame_num'],
                    'audio length': audio.shape[0] / audio_rate
                }
            )
    if len(broken_list) > 0:
        write_csv(broken_list, os.path.join(split_path, 'broken.csv'))
    else:
        tqdm.write('None')
    tqdm.write(f'Sample rate: max = {np.max(sample_rate_list)}, min = {np.min(sample_rate_list)}')



def create_list_for_video(args, name, video_list):
    # import pdb; pdb.set_trace()
    sample_list = []
    if args.split_by_video:
        clip_list = []
        for video in video_list:
            temp = glob.glob(f'{video}/*')
            temp.sort()
            clip_list += temp
        new_video_list = clip_list
    else:
        new_video_list = video_list

    for video in tqdm(new_video_list):
        path = os.path.join('./data/DemoVideo', video)
        meta_path = os.path.join(video, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        frame_num = meta_dict['frame_num']
        frame_rate = meta_dict['frame_rate']
        # import pdb; pdb.set_trace()
        audio_path = os.path.join(video, 'audio', 'audio.wav')
        audio, audio_rate = sf.read(audio_path, dtype='int16')
        cond_2 = not meta_dict['ifstereo']
        cond_3 = np.abs(audio.shape[0] / audio_rate - meta_dict['frame_num'] / meta_dict['frame_rate'] ) > 0.1

        if cond_2 or cond_3:
            continue


        if name == 'train':
            sample_list.append({'path': path})
        elif name == 'val':
            # start_times = np.random.choice(frame_num - frame_rate * 3, 5, replace=False)
            start_times = np.random.choice(frame_num - frame_rate * 3, 10, replace=False)

            for i in start_times:
                sample = {
                    'path': path,
                    'start_time': i
                }
                sample_list.append(sample)
        elif name == 'test':
            start_times = np.random.choice(frame_num - frame_rate * 3, 8, replace=False)
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

    read_path = f'ProcessedData/{args.dataset_name}'
    split_path = f'./data-split/{args.dataset_name}'
    if args.type != '':
        split_path = os.path.join(split_path, args.type)

    os.makedirs(split_path, exist_ok=True)

    data_list = glob.glob(f'{read_path}/*')
    data_list.sort()
    if not args.unshuffle:
        random.shuffle(data_list)

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
    # video_dict = get_download_list(args)

    csv_zip = zip(['train', 'val', 'test'], [train_list, valid_list, test_list])
    for name, video_list in tqdm(csv_zip):
        print(name, len(video_list))
        if len(video_list) == 0:
            continue

        sample_list = create_list_for_video(args, name, video_list)
        if len(video_list) == len(data_list):
            name = 'vis'
        csv_name = f'{split_path}/{name}.csv'
        print(csv_name)
        write_csv(sample_list, csv_name)

# python create-csv.py --data_split='1:0:0'

if __name__ == "__main__":
    args = parser.parse_args()
    if args.exam:
        exam(args)
    else:
        main(args)