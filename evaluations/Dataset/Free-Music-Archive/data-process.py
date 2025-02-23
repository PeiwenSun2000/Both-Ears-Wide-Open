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
parser.add_argument('--split', type=int, default=-1, help='i split of audios to process')
parser.add_argument('--total', type=int, default=15, help='total splits')



def get_audio(read_path, save_path, sample_rate=16000):
    if not os.path.exists(save_path):
        command = f"ffmpeg -v quiet -y -i \"{read_path}\" -acodec pcm_u8 -ar {sample_rate} \"{save_path}\""
        os.system(command)
    
    try: 
        audio, audio_rate = sf.read(save_path, dtype='float32')
    except (RuntimeError, TypeError, NameError):
        return None

    ifstereo = (len(audio.shape) == 2)
    audio_info = {
        'audio_sample_rate': audio_rate,
        'audio_length': audio.shape[0] / audio_rate,
        'ifstereo': ifstereo
    }
    return audio_info

def get_meta(audio, json_path, audio_info):
    meta_dict = audio_info
    with open(json_path, 'w') as fp:
        json.dump(meta_dict, fp, sort_keys=False, indent=4)



def main():
    # import pdb; pdb.set_trace()
    args = parser.parse_args()

    audio_root = 'RawData'
    out_root = 'ProcessedData'
    os.makedirs(out_root, exist_ok=True)
    
    audio_list = glob.glob(f'{audio_root}/*/*.mp3')
    audio_list.sort()

    audio_list = audio_list[int(args.split / args.total * len(audio_list)): int((args.split+1) / args.total * len(audio_list))]
    
    for audio in tqdm(audio_list, desc=f'Audio Processing ID = {str(args.split).zfill(2)}'):
        save_path = audio.split('/')
        save_path[0] = out_root
        save_path[-1] = save_path[-1][:-4]
        save_folder = os.path.join(*save_path)
        os.makedirs(save_folder, exist_ok=True)
        audio_path = os.path.join(save_folder, 'audio.wav')
        meta_path = os.path.join(save_folder, 'meta.json')

        if not os.path.exists(meta_path):
            # import pdb; pdb.set_trace()
            # audio
            audio_info = get_audio(audio, audio_path)

            if audio_info is None:
                shutil.rmtree(save_folder)
                continue
            # meta data
            get_meta(audio, meta_path, audio_info)

        # tqdm.write(f'{audio} is Finished!')
    


if __name__ == "__main__":
    main()
