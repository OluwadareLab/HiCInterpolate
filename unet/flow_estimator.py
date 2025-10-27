import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F


def generate_synthetic_hic(size=256, n_tads=5, loop_prob=0.3, seed=0):
    rng = np.random.default_rng(seed)
    hic = np.zeros((size, size), dtype=np.float32)

    # --- Compartment pattern (A/B)
    x = np.linspace(0, np.pi*2, size)
    compartment = 0.15 * (np.outer(np.sin(x), np.sin(x)) + 1)
    hic += compartment

    # --- TADs (triangular domains along diagonal)
    tad_size = size // n_tads
    for i in range(n_tads):
        start = i * tad_size
        end = min((i+1)*tad_size, size)
        block = np.tril(np.ones((end-start, end-start)))  # triangular TAD
        amp = rng.uniform(0.6, 1.0)
        hic[start:end, start:end] += amp * block

    # --- Loops (bright contacts connecting regions)
    for i in range(n_tads-1):
        if rng.random() < loop_prob:
            x1 = int((i+1)*tad_size - rng.uniform(5, 10))
            x2 = int((i+1)*tad_size + rng.uniform(5, 10))
            val = rng.uniform(0.8, 1.2)
            hic[x1, x2] += val
            hic[x2, x1] += val

    # --- Normalize and blur slightly
    hic = cv2.GaussianBlur(hic, (3, 3), 0)
    hic = (hic - hic.min()) / (hic.max() - hic.min() + 1e-8)

    return hic


# Generate two maps: second one with TAD shifts + loops differences
hic1 = generate_synthetic_hic(seed=0)
# Slight genome rearrangement: small shift + loop pattern change
M = np.float32([[1, 0, 2.0], [0, 1, -2.0]])
hic2 = cv2.warpAffine(
    hic1, M, (hic1.shape[1], hic1.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
hic2 += 0.05*np.random.randn(*hic2.shape)
hic2 = np.clip(hic2, 0, 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Synthetic Hi-C Map 1")
plt.imshow(hic1, cmap='magma')
plt.subplot(1, 2, 2)
plt.title("Synthetic Hi-C Map 2 (shifted)")
plt.imshow(hic2, cmap='magma')
plt.tight_layout()
plt.savefig(
    f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/unet/output/hic_map.png", dpi=300, format='png')
plt.close()


# Use OpenCV dense flow (Farneback) to measure displacement
flow12 = cv2.calcOpticalFlowFarneback(hic1, hic2, None,
                                      pyr_scale=0.5, levels=3, winsize=15,
                                      iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
flow21 = cv2.calcOpticalFlowFarneback(hic2, hic1, None,
                                      pyr_scale=0.5, levels=3, winsize=15,
                                      iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

u, v = flow12[..., 0], flow12[..., 1]
mag = np.sqrt(u**2 + v**2)
ang = np.arctan2(v, u)

# Convert flow to RGB visualization
hsv = np.zeros((hic1.shape[0], hic1.shape[1], 3), dtype=np.uint8)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 1] = 255
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Hi-C Map 1")
plt.imshow(hic1, cmap='magma')
plt.subplot(1, 3, 2)
plt.title("Hi-C Map 2")
plt.imshow(hic2, cmap='magma')
plt.subplot(1, 3, 3)
plt.title("Optical Flow (color)")
plt.imshow(flow_rgb)
plt.tight_layout()
plt.savefig(
    f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/unet/output/flow_map.png", dpi=300, format='png')
plt.close()

print(f"Mean displacement magnitude: {mag.mean():.3f} px")
