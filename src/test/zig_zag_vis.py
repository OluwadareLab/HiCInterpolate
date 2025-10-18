import numpy as np
import matplotlib.pyplot as plt


def zigzag_patches_coords(H, W, patch_size=64, stride=64):
    num_rows = (H - patch_size) // stride + 1
    num_cols = (W - patch_size) // stride + 1
    coords = []
    for i in range(num_rows):
        if i % 2 == 0:
            cols = range(num_cols)
        else:
            cols = reversed(range(num_cols))
        for j in cols:
            r = i * stride
            c = j * stride
            coords.append((r, c))
    return coords, num_rows, num_cols


# Example parameters
H = W = 256
patch_size = 64
coords, n_rows, n_cols = zigzag_patches_coords(H, W, patch_size)

# Create order matrix
order_matrix = np.zeros((n_rows, n_cols))
for k, (r, c) in enumerate(coords):
    order_matrix[r // patch_size, c // patch_size] = k + 1

# Visualization
plt.figure(figsize=(6, 6))
plt.imshow(order_matrix, cmap='viridis', origin='upper')
for i in range(n_rows):
    for j in range(n_cols):
        plt.text(j, i, f'{int(order_matrix[i,j])}',
                 ha='center', va='center', color='white', fontsize=12)
plt.title('Zig-Zag Patch Traversal Order (64×64 patches on 256×256 matrix)')
plt.xlabel('Patch column index')
plt.ylabel('Patch row index')
plt.colorbar(label='Traversal order')
plt.tight_layout()
plt.savefig(f"zig_zag_vis.jpg", dpi=300, format='jpg')
plt.close()
