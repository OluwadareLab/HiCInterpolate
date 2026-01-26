import math
import numpy as np
import torch

def reconstruct_matrix(pred_list, patch_size, h, w):
    n = int(math.ceil(h / patch_size) * patch_size)
    n_patches_per_side = n // patch_size

    patches = np.array([pred.squeeze().cpu().numpy()
                       for batch in pred_list for pred in batch])
    reconstructed = patches.reshape(n_patches_per_side, n_patches_per_side, patch_size, patch_size) \
        .transpose(0, 2, 1, 3) \
        .reshape(n, n)
    pred = reconstructed[:h, :w]
    return pred