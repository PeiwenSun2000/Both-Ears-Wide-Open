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
import csv
from natsort import natsorted
from PIL import Image
from PIL import ImageFilter


parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--split', type=int, default=-1, help='i split of videos to process')
parser.add_argument('--total', type=int, default=15, help='total splits')
parser.add_argument('--frame_rate', type=int, default=10, help='total splits')
parser.add_argument('--max_sample', type=int, default=-1, help='total splits')



def get_facecrop(args, clip, save_path, frame_path, contents):
    # import pdb; pdb.set_trace()
    clip_name = clip.split('/')[-1][:-4]
    speaker_id = clip.split('/')[-2]
    u_id = '_'.join(clip_name.split('_')[:-2])
    begin_time = float(clip_name.split('_')[-2])
    end_time = float(clip_name.split('_')[-1])

    frame_list = glob.glob(f'{frame_path}/*.jpg')
    frame_list.sort()
    bbox_dict = {}

    num_frame = min(len(contents), len(frame_list))
    for i in range(num_frame):
        content = contents[i][:-2].split(' \t')
        bbox = list(map(float, content[1:]))
        image = Image.open(frame_list[i]).convert('RGB')
        image = np.asarray(image)
        img_h, img_w, _ = image.shape
        X, Y, W, H = int(bbox[0] * img_w), int(bbox[1] * img_h), int(bbox[2] * img_w), int(bbox[3] * img_h)
        X = max(0, X)
        Y = max(0, Y)
        Y2 = min(Y+H, img_h)
        X2 = min(X+W, img_w)
        try: 
            new_image = image[Y: Y2, X: X2, :]
            im = Image.fromarray(new_image)
        except (RuntimeError, TypeError, NameError, ValueError):
            tqdm.write(f"Error: {clip}")
        facecrop_path = os.path.join(save_path, frame_list[i].split('/')[-1])
        im.save(facecrop_path)
        bbox_dict[frame_list[i].split('/')[-1]] = bbox
        
    return {
        'frame_num': num_frame,
        'bbox': bbox_dict,
    }



def get_frame(video_path, save_path, fps=10, resol=480):
    command = f'ffmpeg -v quiet -y -i \"{video_path}\" -f image2 -vf \"fps={fps}\" -qscale:v 3 \"{save_path}\"/frame%04d.jpg'
    os.system(command)
    frame_info = {
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

def read_video_dict(args):
    read_path = 'RawVideos/videos'
    speaker_list = glob.glob(f'{read_path}/*')
    speaker_list.sort()
    if args.max_sample > 0:
        speaker_list = speaker_list[:args.max_sample]
    return speaker_list


def get_meta(clip, json_path, frame_info, audio_info, face_info):
    clip_name = clip.split('/')[-1][:-4]
    speaker_id = clip.split('/')[-2]
    u_id = '_'.join(clip_name.split('_')[:-2])
    video_info = {
        'u_id': u_id,
        'category': speaker_id,
        'begin_time': float(clip_name.split('_')[-2]),
        'end_time': float(clip_name.split('_')[-1])
    }
    
    meta_dict = {**video_info, **frame_info, **audio_info, **face_info}
    with open(json_path, 'w') as fp:
        json.dump(meta_dict, fp, sort_keys=False, indent=4)


def read_annotation_for_uid(args, uid_path):
    txtfile_list = glob.glob(f'{uid_path}/*.txt')
    txtfile_list.sort()
    uid_annotation = {}
    for txt_path in txtfile_list:
        with open(txt_path, 'r') as f:
            contents = f.readlines()
        frame_step = int(25 / args.frame_rate)
        contents = contents[7:]
        contents = contents[::frame_step]
        frame_index = int(contents[0][:-2].split(' \t')[0])
        uid_annotation[str(frame_index)] = contents
    return uid_annotation

def main():
    args = parser.parse_args()
    out_root = f'./ProcessedData'
    os.makedirs(out_root, exist_ok=True)
    
    speaker_list = read_video_dict(args)
    speaker_list = speaker_list[int(args.split / args.total * len(speaker_list)): int((args.split+1) / args.total * len(speaker_list))]

    for speaker in tqdm(speaker_list, desc=f'Video Processing ID = {str(args.split).zfill(2)}'):
        speaker_id = speaker.split('/')[-1]
        current_u_id_lists = os.listdir(f'annotations/{speaker_id}')
        current_u_id_lists.sort()
        for u_id in current_u_id_lists:
            uid_annotation = read_annotation_for_uid(args, f'annotations/{speaker_id}/{u_id}')
            clip_list = glob.glob(f'{speaker}/{u_id}*.mp4')
            clip_list = natsorted(clip_list)
            for clip in clip_list:
                # import pdb; pdb.set_trace()
                # detect whether the annotation exists
                clip_name = clip.split('/')[-1][:-4]
                begin_time = float(clip_name.split('_')[-2])
                time_ind = int(begin_time * 25)
                contents = None
                for frame_ind in uid_annotation.keys():
                    if np.abs(time_ind - int(frame_ind)) < 20:
                        contents = uid_annotation[frame_ind]
                if contents is None:
                    continue

                processed_path = os.path.join(out_root, speaker_id, clip_name)
                frame_path = os.path.join(processed_path, 'frames')
                audio_path = os.path.join(processed_path, 'audio')
                face_path = os.path.join(processed_path, 'faces')

                meta_path = os.path.join(processed_path, 'meta.json')
                os.makedirs(frame_path, exist_ok=True)
                os.makedirs(audio_path, exist_ok=True)
                os.makedirs(face_path, exist_ok=True)

                if not os.path.exists(meta_path):
                    # get video
                    final_video = os.path.join(processed_path, 'video.mp4')
                    shutil.copyfile(clip, final_video)

                    # get frame
                    frame_info = get_frame(clip, frame_path, args.frame_rate)
                    # audio
                    audio_info = get_audio(clip, audio_path)

                    # get face
                    face_info = get_facecrop(args, clip, face_path, frame_path, contents)
                    if audio_info is None or face_info is None:
                        tqdm.write(f'{processed_path} is broken')
                        shutil.rmtree(processed_path)
                        continue
                    # meta data
                    get_meta(clip, meta_path, frame_info, audio_info, face_info)

        tqdm.write(f'{speaker_id} is Finished!')
    


if __name__ == "__main__":
    main()

