import numpy as np
import matplotlib.pyplot as plt

# Simulated Hi-C contact map (size 100x100)
np.random.seed(42)
M = np.random.exponential(scale=5, size=(100, 100))  # sparse, heavy-tailed
M = (M + M.T) / 2  # symmetric
np.fill_diagonal(M, np.random.exponential(
    scale=20, size=100))  # strong diagonal
_CMAP = "hot_r"
plt.imshow(M, cmap=_CMAP)
plt.title("Original Hi-C-like Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"original.png", dpi=300, format='png')
plt.close()


M_log = np.log1p(M)
plt.imshow(M_log, cmap=_CMAP)
plt.title("After log1p() Transformation")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"log1p.png", dpi=300, format='png')
plt.close()

percentile = 99.0
upper = np.percentile(M, percentile)
print(f"7{percentile}th percentile = {upper:.2f}")
M_clip = np.clip(M, 0, upper)
plt.imshow(M_clip, cmap=_CMAP)
plt.title(f"After Clipping ({percentile}th Percentile)")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"clipped.png", dpi=300, format='png')
plt.close()

M_log = np.log1p(M)
upper = np.percentile(M_log, percentile)
M_final = np.clip(M_log, 0, upper)
M_final /= upper  # normalize to [0,1]
plt.imshow(M_final, cmap=_CMAP)
plt.title("log1p + Clip + Normalize [0,1]")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"log_clip.png", dpi=300, format='png')
plt.close()


plt.figure(figsize=(10, 5))
plt.hist(M.ravel(), bins=200, alpha=0.5, label='Original')
plt.hist(M_log.ravel(), bins=200, alpha=0.5, label='log1p')
plt.hist(M_clip.ravel(), bins=200, alpha=0.5, label='Clipped')
plt.hist(M_final.ravel(), bins=200, alpha=0.5, label='log1p+Clipped+Scaled')
plt.legend()
plt.title("Distribution Comparison")
plt.tight_layout()
plt.savefig(f"hist.png", dpi=300, format='png')
plt.close()
