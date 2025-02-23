import os
import pickle
import numpy as np
import torch
import sys
sys.path.append("/data/peiwensun/project/audioldm_eval/audioldm_eval/metrics")
from fad import FrechetAudioDistance
import torch.nn.functional as F
import pandas as pd

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

folder1 = '/data/peiwensun/project/stereocrw/feats_gt'
folder2 = '/data/peiwensun/project/stereocrw/feats_ours'
# folder2 = '/data/peiwensun/project/stereocrw/feats_open'
# folder2 = '/data/peiwensun/project/stereocrw/single_ours'

folder1 = '/data/peiwensun/project/stereocrw/gt_double_feats'
folder2 = '/data/peiwensun/project/stereocrw/open_double_feats'
# folder2 = '/data/peiwensun/project/stereocrw/ours_double_feats'

# folder1 = '/data/peiwensun/project/stereocrw/gt_single_moving_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_single_moving_feats'
# folder2 = '/data/peiwensun/project/stereocrw/ours_single_moving_feats'

folder1 = '/data/peiwensun/project/stereocrw/gt_mix_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_mix_feats'
folder2 = '/data/peiwensun/project/stereocrw/ours_mix_feats'

mse_results = {}

data_list1,data_list2=[],[]
df = pd.read_csv("/data/peiwensun/project/audioldm_eval/triplet_intern.csv",header=0)
folder1 = '/data/peiwensun/project/stereocrw/gt_coco_feats'
folder2 = '/data/peiwensun/project/stereocrw/ours_coco_feats'
mse_results = {}

for dix,filename in df.iterrows():
    pkl_name = '{:012d}.pkl'.format(filename["name_2"])
    # print(pkl_name)
    if pkl_name.endswith('.pkl'):
        file1_path = os.path.join(folder1, filename["audio_name_1"]+".pkl")
        file2_path = os.path.join(folder2, pkl_name)
        if os.path.exists(file2_path):
            data1 = load_pickle(file1_path)
            data2 = load_pickle(file2_path)
            data_list1.append(data1["feat_itds"])
            data_list2.append(data2["feat_itds"])

data_list1 = torch.stack(data_list1, dim=0)
data_list2 = torch.stack(data_list2, dim=0)
# print(data_list1.shape,data_list2.shape)

# data_list1*=10
data_list1 = torch.mean(data_list1.view(data_list1.shape[0], 20, 5, 256), dim=2).cpu().numpy()
# data_list1 = data_list1/np.linalg.norm(data_list1, ord=2, axis=-1, keepdims=True)
data_list1 = data_list1.reshape(data_list1.shape[0], -1)

# data_list2*=10
data_list2 = torch.mean(data_list2.view(data_list2.shape[0], 20, 5, 256), dim=2).cpu().numpy()
# data_list2 = data_list2/np.linalg.norm(data_list2, ord=2, axis=-1, keepdims=True)
# data_list2 = F.normalize(data_list2, p=2, dim=-1)
data_list2 = data_list2.reshape(data_list2.shape[0], -1)


# print(data_list1.shape,data_list2.shape)

fad = FrechetAudioDistance()

mu_background, sigma_background = fad.calculate_embd_statistics(data_list1)
mu_eval, sigma_eval = fad.calculate_embd_statistics(data_list2)

fad_score = fad.calculate_frechet_distance(mu_background, sigma_background, mu_eval, sigma_eval)

print(fad_score*10000)