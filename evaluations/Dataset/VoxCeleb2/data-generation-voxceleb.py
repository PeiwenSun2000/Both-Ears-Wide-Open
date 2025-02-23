"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import itertools
import shutil
import csv
import scipy

import glob
import random
import os
import sys
from tqdm import tqdm
import json
import math
import pyroomacoustics as pra

methods = ["ism", "hybrid"]
sound_speed = 343.0

room_dict = {
    'Room_1': {
        'room_dim': [7, 6, 3],
        'mic_l': [3.4, 1, 1.6],
        'mic_r': [3.7, 1, 1.6]
    }, 
    'Room_2': {
        'room_dim': [4, 7, 2.8],
        'mic_l': [0.2, 3.2, 1.7],
        'mic_r': [0.2, 3.0, 1.7]
    },
    'Room_3': {
        'room_dim': [7, 7, 2.7],
        'mic_l': [3.4, 3.1, 1.5],
        'mic_r': [3.5, 2.9, 1.5]
    }
}

parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[0],
        help="Simulation method to use",
    )
parser.add_argument('--rt60', type=float, default=0.1, required=False)
parser.add_argument('--snr', type=float, default=30, required=False)
parser.add_argument('--setting', type=str, default='', required=True)


def get_sound_file(args):
    # import pdb; pdb.set_trace()
    split_path = 'data-split/Newspeaker_ITD/test.csv'
    samples = []
    speaker_dist = {}
    csv_file = csv.DictReader(open(split_path, 'r'), delimiter=',')
    for row in csv_file:
        samples.append(row)
        speaker = row['speaker']
        if not speaker in speaker_dist.keys():
            speaker_dist[speaker] = []
        speaker_dist[speaker].append(row)
    
    return speaker_dist


def locate_source_by_angle(mic_l, mic_r, angle, distance):
    # import pdb; pdb.set_trace()
    mic_l = np.array(mic_l)
    mic_r = np.array(mic_r)
    center = (mic_l + mic_r) / 2
    vector_lr = mic_r - mic_l
    
    # slope = - vector_lr[0] / vector_lr[1] 
    vector_norm = np.array([-vector_lr[1], vector_lr[0], 0])
    vector_norm = vector_norm / np.linalg.norm(vector_norm)
    cross = np.cross(vector_lr, vector_norm)
    if cross[2] < 0:
        vector_norm = - vector_norm
    if vector_norm[0] == 0:
        theta = 90 if vector_norm[1] > 0 else -90
    else:
        theta = np.arctan(vector_norm[1]/vector_norm[0]) / np.pi * 180
    convert_angle = (theta - angle)/180 * np.pi
    source_loc = np.array([center[0] + distance * np.cos(convert_angle), center[1] + distance * np.sin(convert_angle), center[2]])

    c = sound_speed
    itd = (np.linalg.norm(source_loc - mic_r) - np.linalg.norm(source_loc - mic_l) ) / c 
    return source_loc, itd


def inverse_sabine(rt60, room_dim, c=None):
    """
    Given the desired reverberation time (RT60, i.e. the time for the energy to
    drop by 60 dB), the dimensions of a rectangular room (shoebox), and sound
    speed, computes the energy absorption coefficient and maximum image source
    order needed. The speed of sound used is the package wide default (in
    :py:data:`~pyroomacoustics.parameters.constants`).
    Parameters
    ----------
    rt60: float
        desired RT60 (time it takes to go from full amplitude to 60 db decay) in seconds
    room_dim: list of floats
        list of length 2 or 3 of the room side lengths
    c: float
        speed of sound
    Returns
    -------
    absorption: float
        the energy absorption coefficient to be passed to room constructor
    max_order: int
        the maximum image source order necessary to achieve the desired RT60
    """
    # import pdb; pdb.set_trace()
    if c is None:
        c = sound_speed

    if rt60 == 0:
        e_absorption = 0.999
        max_order = 0
        return e_absorption, max_order

    # finding image sources up to a maximum order creates a (possibly 3d) diamond
    # like pile of (reflected) rooms. now we need to find the image source model order
    # so that reflections at a distance of at least up to ``c * rt60`` are included.
    # one possibility is to find the largest sphere (or circle in 2d) that fits in the
    # diamond. this is what we are doing here.
    R = []
    for l1, l2 in itertools.combinations(room_dim, 2):
        R.append(l1 * l2 / np.sqrt(l1 ** 2 + l2 ** 2))

    V = np.prod(room_dim)  # area (2d) or volume (3d)
    # "surface" computation is diff for 2d and 3d
    if len(room_dim) == 2:
        S = 2 * np.sum(room_dim)
        sab_coef = 12  # the sabine's coefficient needs to be adjusted in 2d
    elif len(room_dim) == 3:
        S = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
        sab_coef = 24

    e_absorption = (
        sab_coef * np.log(10) * V / (c * S * rt60)
    )  # absorption in power (sabine)
    if e_absorption >= 1:
        e_absorption = 0.85
    # the int cast is only needed for python 2.7
    # math.ceil returns int for python 3.5+
    max_order = int(math.ceil(c * rt60 / np.min(R) - 1))

    return e_absorption, max_order



def generate_audio(args, room_dim, rt60_tgt, fs, audios, source_locs, mic_locs, save_path):
    # import pdb; pdb.set_trace()
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    if args.method == "ism":
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
        )
    elif args.method == "hybrid":
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
        )

    # place the source in the room
    room.add_source(source_locs[0], signal=audios[0], delay=0.0)
    room.add_source(source_locs[1], signal=audios[1], delay=0.0)

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate(reference_mic=0, snr=args.snr)
    room.mic_array.to_wav(
        save_path,
        norm=True,
        bitdepth=np.int16,
    )

    # measure the reverberation time
    try: 
        rt60 = room.measure_rt60().reshape(-1)
    except (RuntimeError, TypeError, NameError, ValueError):
        rt60 = [rt60_tgt * 1.4, rt60_tgt * 1.4]
    
    # print("The desired RT60 was {}".format(rt60_tgt))
    # print("The measured RT60 is {}".format(rt60[1, 0]))
    
    meta_data = {
        'Audio Sample Rate': fs,
        'Audio Length': audios[0].shape[0] / fs,
        'Rt60_measured': list(rt60)
    }
    return meta_data


def select_mixture_audio(speaker_dict, speaker1, num_of_each_speaker):
        # import pdb; pdb.set_trace()
    candicates = list(speaker_dict.keys())
    candicates.remove(speaker1)
    new_speakers = np.random.choice(candicates, num_of_each_speaker, replace=False)
    new_speaker_audio_list = [np.random.choice(speaker_dict[i]) for i in new_speakers]
    return new_speaker_audio_list


def generate_frames(args, speaker1, speaker2, save_folder):
    speaker1_read_framepath = os.path.join(*speaker1.split('/')[3:], 'frames')
    speaker2_read_framepath = os.path.join(*speaker2.split('/')[3:], 'frames')

    speaker1_save_framepath = os.path.join(save_folder, 'speaker1_frames')
    speaker2_save_framepath = os.path.join(save_folder, 'speaker2_frames')

    if not os.path.exists(speaker1_save_framepath):
        shutil.copytree(speaker1_read_framepath, speaker1_save_framepath)

    if not os.path.exists(speaker2_save_framepath):
        shutil.copytree(speaker2_read_framepath, speaker2_save_framepath)

    speaker1_read_facepath = os.path.join(*speaker1.split('/')[3:], 'faces')
    speaker2_read_facepath = os.path.join(*speaker2.split('/')[3:], 'faces')

    speaker1_save_facepath = os.path.join(save_folder, 'speaker1_faces')
    speaker2_save_facepath = os.path.join(save_folder, 'speaker2_faces')

    if not os.path.exists(speaker1_save_facepath):
        shutil.copytree(speaker1_read_facepath, speaker1_save_facepath)

    if not os.path.exists(speaker2_save_facepath):
        shutil.copytree(speaker2_read_facepath, speaker2_save_facepath)

    speaker1_meta_path = os.path.join(*speaker1.split('/')[3:], 'meta.json')
    speaker2_meta_path = os.path.join(*speaker2.split('/')[3:], 'meta.json')
    with open(speaker1_meta_path, "r") as f:
        speaker1_meta_dict = json.load(f)
    with open(speaker2_meta_path, "r") as f:
        speaker2_meta_dict = json.load(f)
    frame_rate = speaker1_meta_dict['frame_rate']
    speaker1_id  = speaker1_meta_dict['category']
    speaker2_id  = speaker2_meta_dict['category']

    return {
        'frame_rate': frame_rate,
        'speaker1_id': speaker1_id, 
        'speaker2_id': speaker2_id
    }


def normalize_audio(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    samples[samples > 1.] = 1.
    samples[samples < -1.] = -1.
    return samples 

def create_synthetic_sample(args, room, rt60_tgt, speaker1, speaker2, save_folder):
    # import pdb; pdb.set_trace()

    room_dim = room['room_dim'] # meters
    # define the locations of the microphones
    mic_l = room['mic_l']
    mic_r = room['mic_r']
    mic_locs = np.c_[mic_l, mic_r]
    fs = 16000
    speaker1_audio_path = os.path.join(*speaker1.split('/')[3:], 'audio', 'audio.wav')
    speaker2_audio_path = os.path.join(*speaker2.split('/')[3:], 'audio', 'audio.wav')
    speaker1_audio, speaker1_fs = sf.read(speaker1_audio_path, dtype='float32', always_2d=True)
    speaker2_audio, speaker2_fs = sf.read(speaker2_audio_path, dtype='float32', always_2d=True)
    speaker1_audio = np.mean(speaker1_audio, axis=-1)
    speaker2_audio = np.mean(speaker2_audio, axis=-1)

    if speaker1_fs != fs:
        speaker1_audio = scipy.signal.resample(speaker1_audio, int(speaker1_audio.shape[0] / speaker1_fs * fs), axis=0)
    if speaker2_fs != fs:
        speaker2_audio = scipy.signal.resample(speaker2_audio, int(speaker2_audio.shape[0] / speaker2_fs * fs), axis=0)

    audio_length = min(speaker1_audio.shape[0], speaker2_audio.shape[0])
    speaker1_audio = speaker1_audio[:audio_length]
    speaker2_audio = speaker2_audio[:audio_length]

    speaker1_rms = np.sqrt(np.mean(speaker1_audio ** 2))
    speaker2_audio = normalize_audio(speaker2_audio, desired_rms=speaker1_rms*1.05)

    angles = np.random.choice(np.arange(-90, 91), 2, replace=False).astype(float)
    distance = np.random.uniform(0.5, 3.0)
    speaker1_loc, itd1 = locate_source_by_angle(mic_l, mic_r, angle=angles[0], distance=distance)
    speaker2_loc, itd2 = locate_source_by_angle(mic_l, mic_r, angle=angles[1], distance=distance)

    save_path = os.path.join(save_folder, 'audio.wav')
    audio_meta = generate_audio(args, room_dim, rt60_tgt, fs, (speaker1_audio, speaker2_audio), (speaker1_loc, speaker2_loc), mic_locs, save_path)
    frame_meta = generate_frames(args, speaker1, speaker2, save_folder)
    
    meta_path = os.path.join(save_folder, 'meta.json')
    meta = {
        'Room Dim': room_dim,
        'Rt60_desired': rt60_tgt,
        'Speaker1 Loc': list(speaker1_loc),
        'Speaker2 Loc': list(speaker2_loc),
        'Left mic Loc': mic_l,
        'Right mic Loc': mic_r,
        'Angle': list(angles),
        'Distance': distance,
        'itd1': itd1,
        'itd2': itd2,
        'speaker1_path': os.path.join(*speaker1.split('/')[3:]),
        'speaker2_path': os.path.join(*speaker2.split('/')[3:])
    }
    meta_data = {**meta, **audio_meta, **frame_meta}
    with open(meta_path, 'w') as fp:
        json.dump(meta_data, fp, sort_keys=False, indent=4)


def main():
    args = parser.parse_args()
    if args.setting == 'Easy':
        args.snr = 30
        args.rt60 = 0.1
    elif args.setting == 'Hard':
        args.snr = 10
        args.rt60 = 0.5
    np.random.seed(2021)
    random.seed(2021)

    if args.setting == '':
        save_dir = f'ProcessedData-TDE'
    else: 
        save_dir = f'ProcessedData-TDE/{args.setting}'
    os.makedirs(save_dir, exist_ok=True)
    speaker_dict = get_sound_file(args)
    # The desired reverberation time and dimensions of the room
    rt60_tgt = args.rt60 / 1.4 # seconds
    count = 0
    num_of_each_speaker = 4
    for ind, room_key in enumerate(room_dict):
        # import pdb; pdb.set_trace()
        for speaker1 in tqdm(speaker_dict.keys(), total=len(speaker_dict.keys()), desc=f'Room {ind}'):
            if len(speaker_dict[speaker1]) < num_of_each_speaker:
                continue
            speaker1_list = np.random.choice(speaker_dict[speaker1], num_of_each_speaker, replace=False)
            speaker2_list = select_mixture_audio(speaker_dict, speaker1, num_of_each_speaker)
            for i in range(num_of_each_speaker):
                # import pdb; pdb.set_trace()
                room = room_dict[room_key]
                save_folder = os.path.join(save_dir, f'Room-{str(ind).zfill(3)}', f'sample-{str(count).zfill(4)}')
                os.makedirs(save_folder, exist_ok=True)
                create_synthetic_sample(args, room, rt60_tgt, speaker1_list[i]['path'], speaker2_list[i]['path'], save_folder)
                count += 1
            
# python data-generation-voxceleb.py --setting='Easy'
# python data-generation-voxceleb.py --setting='Hard'


if __name__ == "__main__":
    main()