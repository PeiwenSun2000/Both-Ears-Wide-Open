import os
import pickle
import numpy as np
import pandas as pd
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

folder1 = '/data/peiwensun/project/stereocrw/single_gt'
folder2 = '/data/peiwensun/project/stereocrw/single_open'
# folder2 = '/data/peiwensun/project/stereocrw/single_ours'

folder1 = '/data/peiwensun/project/stereocrw/gt_double_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_double_feats'
folder2 = '/data/peiwensun/project/stereocrw/ours_double_feats'

# folder1 = '/data/peiwensun/project/stereocrw/gt_single_moving_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_single_moving_feats'
# folder2 = '/data/peiwensun/project/stereocrw/ours_single_moving_feats'

folder1 = '/data/peiwensun/project/stereocrw/gt_mix_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_mix_feats'
folder2 = '/data/peiwensun/project/stereocrw/ours_mix_feats'

df = pd.read_csv("/data/peiwensun/project/audioldm_eval/triplet_intern.csv",header=0)
folder1 = '/data/peiwensun/project/stereocrw/gt_coco_feats'
folder2 = '/data/peiwensun/project/stereocrw/s2s_coco_feats'
mse_results = {}

for dix,filename in df.iterrows():
    pkl_name = '{:012d}.pkl'.format(filename["name_2"])
    print(pkl_name)
    if pkl_name.endswith('.pkl'):
        file1_path = os.path.join(folder1, filename["audio_name_1"]+".pkl")
        file2_path = os.path.join(folder2, pkl_name)
        if os.path.exists(file2_path):
            data1 = load_pickle(file1_path)
            data2 = load_pickle(file2_path)
            feat1 = data1["crw_itds"].mean()
            # feat2 = data2["crw_itds"].mean()
            # feat1 = data1["baseline_itds"].mean()
            feat2 = 0
            print(feat1,feat2)
            # feat2 = 0
            # quit()
            # Assuming data1 and data2 are numpy arrays or compatible types
            mse = calculate_mse(feat1, feat2)
            mse_results['{:012d}'.format(filename["name_2"])+filename["audio_name_1"]] = mse
# Print the MSE results
# for file, mse in mse_results.items():
#     print(f'MSE for {file}: {mse}')
print("Num Samples:",len(mse_results.values()))
print("Overll mean:",1000*sum(mse_results.values()) / len(mse_results.values()))
