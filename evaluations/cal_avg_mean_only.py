import os
import pickle
import numpy as np
from tqdm import tqdm

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

folder1 = '/data/peiwensun/project/stereocrw/gt_double_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_double_feats'
folder2 = '/data/peiwensun/project/stereocrw/ours_double_feats'

# folder1 = '/data/peiwensun/project/stereocrw/gt_single_moving_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_single_moving_feats'
# folder2 = '/data/peiwensun/project/stereocrw/ours_single_moving_feats'

folder1 = '/data/peiwensun/project/stereocrw/gt_mix_feats'
# folder1 = '/data/peiwensun/project/stereocrw/open_mix_feats'
# folder1 = '/data/peiwensun/project/stereocrw/ours_mix_feats'
# folder2 = '/data/peiwensun/project/stereocrw/ours_mix_feats'

# folder1 = '/data/peiwensun/project/stereocrw/gt_real_world_feats'
# folder2 = '/data/peiwensun/project/stereocrw/open_real_world_feats'
# folder2 = '/data/peiwensun/project/stereocrw/ours_real_world_feats'

# folder1 = '/data/peiwensun/project/stereocrw/s2s_eval_feats'
# folder1 = '/data/peiwensun/project/stereocrw/s2s_coco_feats'
# folder1 = '/data/peiwensun/project/stereocrw/observe_open_feats'
# folder1 = '/data/peiwensun/project/stereocrw/observe_feats'
# folder1 = '/data/peiwensun/project/stereocrw/observe_posi_feats'
# folder1 = '/data/peiwensun/project/stereocrw/gt_coco_feats'
# folder1 = '/data/peiwensun/project/stereocrw/ours_coco_feats'
mse_results = []

def calculate_variance(numbers):
    # Step 1: Calculate the mean of the numbers
    mean = sum(numbers) / len(numbers)
    
    # Step 2: Calculate the squared differences from the mean
    squared_diffs = [(x - mean) ** 2 for x in numbers]
    
    # Step 3: Calculate the variance
    variance = sum(squared_diffs) / len(numbers)  # Population variance
    # If you need sample variance, use (len(numbers) - 1) instead of len(numbers)
    # variance = sum(squared_diffs) / (len(numbers) - 1)
    
    return variance

for filename in tqdm(os.listdir(folder1)):
    if filename.endswith('.pkl'):
        file1_path = os.path.join(folder1, filename)
        data1 = load_pickle(file1_path)
        feat1 = data1["crw_itds"].mean()
        # feat1 = abs(data1["crw_itds"]).mean()
        if -0.01<feat1<0.01:
            mse_results.append(100*feat1)
        print(feat1)
print(calculate_variance(mse_results))
# print("Overll mean:",100*sum(mse_results) / len(mse_results))
