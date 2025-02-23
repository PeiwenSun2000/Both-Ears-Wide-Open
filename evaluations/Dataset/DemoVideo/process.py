import pdb
import subprocess
import argparse
import re
import cv2
import sys
import os
import glob
import json
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm
import soundfile as sf
import shutil


parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--split', type=int, default=0, help='i split of videos to process')
parser.add_argument('--total', type=int, default=1, help='total splits')
parser.add_argument('--frame_rate', type=int, default=10, help='total splits')
parser.add_argument('--dataset_name', default='', type=str)




def get_frame(video_path, save_path, fps=10, resol=480):
    command = f'ffmpeg -v quiet -y -i \"{video_path}\" -f image2 -vf \"scale=-1:{resol},fps={fps}\" -qscale:v 3 \"{save_path}\"/frame%04d.jpg'
    os.system(command)
    frame_info = {
        'frame_num': len(os.listdir(save_path)),
        'frame_rate': fps
    }
    return frame_info


def get_audio(video_path, save_path):
    audio_path = os.path.join(save_path, 'audio.wav')
    if not os.path.exists(audio_path):
        command = f"ffmpeg -v quiet -y -i \"{video_path}\" \"{audio_path}\""
        os.system(command)

    try: 
        audio, audio_rate = sf.read(audio_path, start=0, stop=10000, dtype='int16')
    except (RuntimeError, TypeError, NameError):
        return None

    # audio, audio_rate = sf.read(audio_path, start=0, stop=10000, dtype='int16')
    ifstereo = (len(audio.shape) == 2)
    audio_info = {
        'audio_sample_rate': audio_rate,
        'ifstereo': ifstereo
    }
    return audio_info

def get_meta(clip, json_path, frame_info, audio_info):
    video_info = {}
    
    meta_dict = {**video_info, **frame_info, **audio_info}
    with open(json_path, 'w') as fp:
        json.dump(meta_dict, fp, sort_keys=False, indent=4)


def main():
    args = parser.parse_args()
    video_root = f'./RawVideos/{args.dataset_name}'
    out_root = f'./ProcessedData/{args.dataset_name}'
    os.makedirs(out_root, exist_ok=True)
    
    video_list = glob.glob(f'{video_root}/*')
    video_list.sort()
    print(video_list)
    video_list = video_list[int(args.split / args.total * len(video_list)): int((args.split+1) / args.total * len(video_list))]
    print(video_list)
    for video in tqdm(video_list, desc=f'Video Processing ID = {str(args.split).zfill(2)}'):
        # import pdb; pdb.set_trace()
        video_name = video.split('/')[-1][:-4]
        processed_path = os.path.join(out_root, video_name)
        frame_path = os.path.join(processed_path, 'frames')
        audio_path = os.path.join(processed_path, 'audio')
        meta_path = os.path.join(processed_path, 'meta.json')
        os.makedirs(frame_path, exist_ok=True)
        os.makedirs(audio_path, exist_ok=True)

        if not os.path.exists(meta_path):
            frame_info = get_frame(video, frame_path, args.frame_rate)
            # audio
            audio_info = get_audio(video, audio_path)
            
            if audio_info is None:
                tqdm.write(f'{processed_path} is broken')
                shutil.rmtree(processed_path)
                continue
            # meta data
            get_meta(video, meta_path, frame_info, audio_info)

        tqdm.write(f'{video_name} is Finished!')
    


if __name__ == "__main__":
    main()

