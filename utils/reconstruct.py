import numpy as np
import torch
import math

def reconstruct_from_batches(batch_predictions, h, w, patch_size, stride):
    """
    Reconstructs an n x n matrix from a list of batch predictions.
    
    Args:
        batch_predictions (list): List of Tensors/Arrays from the model (shape: [B, 1, P, P] or [B, P, P]).
        h (int): Original height of the matrix (number of bins).
        w (int): Original width of the matrix (number of bins).
        patch_size (int): Size of the square patches (e.g., 64).
        stride (int): The stride used during patch extraction (bin_inc).
        
    Returns:
        np.ndarray: The reconstructed h x w matrix.
    """
    # 1. Flatten batches into a single list of 2D numpy patches
    all_patches = []
    for batch in batch_predictions:
        if torch.is_tensor(batch):
            batch = batch.detach().cpu().numpy()
        
        # Handle potential channel dimension [Batch, 1, P, P] -> [Batch, P, P]
        if batch.ndim == 4:
            batch = np.squeeze(batch, axis=1)
        
        for i in range(batch.shape[0]):
            all_patches.append(batch[i])

    # 2. Determine the internal dimensions used during padding
    # Based on inference_cool_to_square_matrix.py padding logic
    pad_h = int(math.ceil(h / patch_size) * patch_size) if stride == patch_size else h
    pad_w = int(math.ceil(w / patch_size) * patch_size) if stride == patch_size else w
    
    # 3. Initialize target matrix and weight matrix (for averaging overlaps)
    recon_matrix = np.zeros((pad_h, pad_w), dtype=np.float32)
    weight_matrix = np.zeros((pad_h, pad_w), dtype=np.float32)
    
    patch_idx = 0
    # Standard grid traversal used in generate_patch
    for r in range(0, pad_h - patch_size + 1, stride):
        for c in range(0, pad_w - patch_size + 1, stride):
            if patch_idx >= len(all_patches):
                break
            
            patch = all_patches[patch_idx]
            recon_matrix[r:r+patch_size, c:c+patch_size] += patch
            weight_matrix[r:r+patch_size, c:c+patch_size] += 1.0
            patch_idx += 1
            
    # 4. Average overlapping regions and crop to original size
    with np.errstate(divide='ignore', invalid='ignore'):
        final_matrix = recon_matrix / weight_matrix
        final_matrix = np.nan_to_num(final_matrix) # Replace NaNs with 0
        
    return final_matrix[:h, :w]