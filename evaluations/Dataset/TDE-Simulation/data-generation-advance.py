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
parser.add_argument('--rt60', type=float, default=0.1, required=True)


def get_sound_file(args):
    # import pdb; pdb.set_trace()
    read_path = './TIMIT/data'
    read_list = glob.glob(f"{read_path}/*/*/*/*.wav")
    read_list.sort()
    random.seed(2021)
    random.shuffle(read_list)
    test_list = read_list
    return test_list

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



def generate_audio(args, room_dim, rt60_tgt, wav_path, source_loc, mic_locs, save_path):
    # import pdb; pdb.set_trace()
    audio_path = os.path.join(save_path, 'audio.wav')
    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    audio, fs = sf.read(wav_path, dtype='int16')
    # fs, audio = wavfile.read(wav_path)

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
    room.add_source(source_loc, signal=audio, delay=0.0)
    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()
    room.mic_array.to_wav(
        audio_path,
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
        'Audio Length': audio.shape[0] / fs,
        'Rt60_measured': list(rt60)
    }
    return meta_data


def main():
    args = parser.parse_args()
    save_dir = f'ProcessedData/RT60_{args.rt60}'
    sound_list = get_sound_file(args)
    # The desired reverberation time and dimensions of the room
    rt60_tgt = args.rt60 / 1.4 # seconds
    count = 0
    for ind, room_key in enumerate(room_dict):
        # import pdb; pdb.set_trace()
        num_sound = int(len(sound_list) // len(room_dict))
        wav_list = sound_list[int(ind * num_sound): int((ind + 1) * num_sound)]
        room = room_dict[room_key]
        room_dim = room['room_dim'] # meters

        # define the locations of the microphones
        mic_l = room['mic_l']
        mic_r = room['mic_r']
        mic_locs = np.c_[mic_l, mic_r]
        np.random.seed(2021)
        for ind in tqdm(range(len(wav_list))):
            # import pdb; pdb.set_trace()
            wav_path = wav_list[ind]
            save_path = os.path.join(save_dir, f'sample-{str(count).zfill(4)}')
            os.makedirs(save_path, exist_ok=True)
            angle = np.random.randint(-90, 91)
            distance = np.random.uniform(0.5, 3.0)
            source_loc, itd = locate_source_by_angle(mic_l, mic_r, angle=angle, distance=distance)
            audio_meta = generate_audio(args, room_dim, rt60_tgt, wav_path, source_loc, mic_locs, save_path)

            meta_path = os.path.join(save_path, 'meta.json')
            meta = {
                'Room Dim': room_dim,
                'Rt60_desired': rt60_tgt,
                'Source Loc': list(source_loc),
                'Left mic Loc': mic_l,
                'Right mic Loc': mic_r,
                'Angle': angle,
                'Distance': distance,
                'itd': itd
            }
            meta_data = {**meta, **audio_meta}
            count += 1
            with open(meta_path, 'w') as fp:
                json.dump(meta_data, fp, sort_keys=False, indent=4)




if __name__ == "__main__":
    main()