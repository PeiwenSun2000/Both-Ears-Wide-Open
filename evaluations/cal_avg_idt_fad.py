import os
import pickle
import numpy as np
import torch
import sys
sys.path.append("/data/peiwensun/project/audioldm_eval/audioldm_eval/metrics")
from fad import FrechetAudioDistance
import torch.nn.functional as F

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

folder1 = '/data/peiwensun/project/stereocrw/temp_1'
folder2 = '/data/peiwensun/project/stereocrw/temp_2'

mse_results = {}
def softmax_normalize(x):
    return F.softmax(torch.tensor(x), dim=-1)
data_list1, data_list2 = [], []
for filename in os.listdir(folder2):
    if filename.endswith('.pkl'):
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        # print(file1_path,file2_path)
        if os.path.exists(file1_path):
            # Apply softmax normalization to feat_itds
            data1 = load_pickle(file1_path)
            # data1["feat_itds"]=softmax_normalize(data1["feat_itds"])
            data2 = load_pickle(file2_path)
            # data2["feat_itds"]=softmax_normalize(data2["feat_itds"])
            data_list1.append(data1["feat_itds"])
            data_list2.append(data2["feat_itds"])

data_list1 = torch.stack(data_list1, dim=0)
data_list2 = torch.stack(data_list2, dim=0)
# data_list2 = torch.flip(data_list2, dims=[1])

data_list1 = torch.mean(data_list1.view(data_list1.shape[0], 20, 5, 256), dim=2).cpu().numpy()
data_list1 = data_list1.reshape(data_list1.shape[0], -1)
data_list2 = torch.mean(data_list2.view(data_list2.shape[0], 20, 5, 256), dim=2).cpu().numpy()
data_list2 = data_list2.reshape(data_list2.shape[0], -1)

fad = FrechetAudioDistance()

mu_background, sigma_background = fad.calculate_embd_statistics(data_list1)
mu_eval, sigma_eval = fad.calculate_embd_statistics(data_list2)

fad_score = fad.calculate_frechet_distance(mu_background, sigma_background, mu_eval, sigma_eval)

print("[info] fsad_score:", fad_score * 1000)
