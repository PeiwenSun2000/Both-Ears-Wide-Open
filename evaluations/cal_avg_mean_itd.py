import os
import pickle
import numpy as np

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

folder1 = '/data/peiwensun/project/stereocrw/temp_1'
folder2 = '/data/peiwensun/project/stereocrw/temp_2'

def normalize(distribution):
    """
    Normalize the input list to a probability distribution.
    :param distribution: Array or list
    :return: Normalized probability distribution
    """
    distribution = np.array(distribution)
    total = np.sum(distribution)
    if total == 0:
        return np.ones_like(distribution) / len(distribution)
    return distribution / total

def kl_divergence(p, q):
    """
    Calculate the KL divergence between two probability distributions.
    :param p: Probability distribution P, in array form
    :param q: Probability distribution Q, in array form
    :return: KL divergence value
    """
    p = normalize(p)
    q = normalize(q)
    epsilon = 1e-10
    q = np.clip(q, epsilon, 1.0)
    p = np.clip(p, epsilon, 1.0)

    return np.sum(p * np.log(p / q))

mse_results = {}
variance1, variance2 = [], []
mse_results_crw = {}
mse_results_gcc = {}
variance1_crw, variance2_crw = [], []
variance1_gcc, variance2_gcc = [], []

for filename in os.listdir(folder1):
    if filename.endswith('.pkl'):
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        if os.path.exists(file2_path):
            data1 = load_pickle(file1_path)
            data2 = load_pickle(file2_path)
            
            # Calculate for CRW mode
            feat1_crw = data1["crw_itds"].mean()
            feat2_crw = data2["crw_itds"].mean()
            variance1_crw.append(feat1_crw)
            variance2_crw.append(feat2_crw)
            # print(feat1_crw, feat2_crw)
            mse_crw = calculate_mse(feat1_crw, feat2_crw)
            mse_results_crw[filename] = mse_crw
            
            # Calculate for GCC mode
            feat1_gcc = data1["baseline_itds"].mean()
            feat2_gcc = data2["baseline_itds"].mean()
            variance1_gcc.append(feat1_gcc)
            variance2_gcc.append(feat2_gcc)
            mse_gcc = calculate_mse(feat1_gcc, feat2_gcc)
            mse_results_gcc[filename] = mse_gcc

# Results for CRW mode
variance2_crw = [sum(variance1_crw) / len(variance1_crw)] * len(variance1_crw)
print("\nResults for CRW mode:")
print("Num Samples:", len(mse_results_crw.values()))
print("KL-divergence", kl_divergence(variance1_crw, variance2_crw))
print("[info] Overall ITD MSE:", 1000 * sum(mse_results_crw.values()) / len(mse_results_crw.values()))

# Results for GCC mode
variance2_gcc = [sum(variance1_gcc) / len(variance1_gcc)] * len(variance1_gcc)
print("\nResults for GCC mode:")
print("Num Samples:", len(mse_results_gcc.values()))
print("KL-divergence", kl_divergence(variance1_gcc, variance2_gcc))
print("[info] Overall ITD MSE:", 1000 * sum(mse_results_gcc.values()) / len(mse_results_gcc.values()))
