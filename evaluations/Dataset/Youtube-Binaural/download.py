import os
import csv
import sys
import numpy as np 
import argparse
from time import strftime
from time import gmtime
from tqdm import tqdm 
import errno
import inspect
import json

parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--split', type=int, default=-1, help='i split of videos to process')
parser.add_argument('--total', type=int, default=15, help='total splits')
parser.add_argument('--dataset_name', default='', type=str)
parser.add_argument('--data_csv', default='', type=str)



def get_download_list(args):
    csv_path = f'./data-info/{args.data_csv}.csv'
    if not os.path.exists(csv_path):
        tqdm.write('CSV file is not exist!')
        exit()

    video_dict = {}
    csv_file = csv.reader(open(csv_path, 'r'), delimiter=',')
    next(csv_file)
    for row in csv_file:
        clip_dict = {
            'u_id': row[0],
            'start_time': int(row[1]),
            'end_time': int(row[2])
        }
        if not row[0] in video_dict.keys():
            video_dict[row[0]] = []
        video_dict[row[0]].append(clip_dict)
    # import pdb; pdb.set_trace()
    return video_dict


def download_video(args):
    # fps = 10 
    video_dict = get_download_list(args)
    key_list = list(video_dict.keys())
    key_list.sort()
    key_list = key_list[int(args.split / args.total * len(key_list)): int((args.split+1) / args.total * len(key_list))]
    
    root = f'./RawVideos/{args.dataset_name}'
    os.makedirs(root, exist_ok=True)

    record_dir = f'./download-list/{args.dataset_name}'
    os.makedirs(record_dir, exist_ok=True)
    json_file = f'{record_dir}/download_list_{str(args.split).zfill(2)}.json'
    if not os.path.exists(json_file):
        with open(json_file, "w") as fp:
            json.dump([], fp, sort_keys=False, indent=2)

    for url in tqdm(key_list, desc=f'Video Processing ID = {str(args.split).zfill(2)}'):
        # import pdb; pdb.set_trace()
        with open(json_file, "r") as f:
            download_list = json.load(f)
        
        if url in download_list:
            continue

        # download the whole video
        video_folder = os.path.join(root, url)
        os.makedirs(video_folder, exist_ok=True)

        video_file = os.path.join(video_folder, f'{url}.mp4')
        dl_command = f"yt-dlp -q -w -c -f mp4 -o \"{video_file}\" -- \"{url}\""
        os.system(dl_command)
        
        # to continue if youtube video is unavailable 
        if not os.path.exists(video_file):
            tqdm.write('{} is broken for some reasons'.format(url))
            os.system(f'rm -rf {video_folder}')
            # os.system(f'rm -f {video_folder}/{url}*.part')
            continue

        time_list = video_dict[url]
        for slot in time_list:
            begin, end = slot['start_time'], slot['end_time']
            duration = end - begin
            clip_name = f'{url}_{str(begin).zfill(4)}_{str(end).zfill(4)}.mp4'
            clip_path = os.path.join(video_folder, clip_name)

            begin_time = strftime("%H:%M:%S", gmtime(begin))
            end_time = strftime("%H:%M:%S", gmtime(end))

            # if not os.path.exists(clip_path):
            command = f'ffmpeg -v quiet -y -ss {begin_time} -to {end_time} -i \"{video_file}\"  \"{clip_path}\"'
            os.system(command)


        download_list.append(url)
        with open(json_file, "w") as fp:
            json.dump(download_list, fp, sort_keys=True, indent=2)
        os.remove(video_file)
        tqdm.write('{} is Finished!'.format(url))


# python ASMR-download.py --split=0 --total=1
if __name__ == "__main__":
    args = parser.parse_args()
    download_video(args)
