from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cupy as cp
import torch
import numpy as np
import _4DMax.Utils.util as ut
import _4DMax.Utils.movement as mv
import _4DMax.Utils.likelihood as li

DATA_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset'
DICT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset/test'
OUTPUT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/final_output/Plot_4DMax'

RESOLUTIONS = [25000]
PATCHES = [64]
CHROMOSOMES = {
    "human": ["10", "11"]
}
DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                [
                    ["4DNFI7T93SHL_dmso_control_30m",
                     "4DNFICF2Z2TG_dmso_control_60m",
                     "4DNFILL624WG_dmso_control_90m"]
                ]
            }
        },
        "dtag": {
            "v1": {
                "triplets":
                [

                    ["4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m"]
                ]
            }
        }
    }
}

columns = ["Resolution", "Patch Size", "organism", "sample", "condition", "Timeframe",  "chromosome", "PSNR",
           "SSIM", "SCC", "GenomeDISCO", "HiCRep", "LPIPS"]

CSV_FILENAME = f"{OUTPUT_DIR}/4DMax_results.csv"

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300
CMAP_ = mcolors.LinearSegmentedColormap.from_list(
    "juicebox", ["#FFFFFF", "#FFAAAA", "#FF5555", "#FF0000", "#B30000"], N=256
)


def draw_hic_comparison_one(num_examples, target: torch.Tensor, title, file, count):
    matrix = np.log1p(target.squeeze().detach().cpu().numpy())
    _min = np.min(matrix)
    _max = np.max(matrix)
    matrix = (matrix - _min) / (_max - _min + 1e-10)  # Normalize to [0, 1]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap=CMAP_)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{file.removesuffix('.png')}_{title}_{count}.png",
                dpi=300, format='png')
    plt.close()


def get_sparse_matrix(matrix):
    rows, cols = np.where(matrix > 0)
    values = matrix[rows, cols]

    sparse_matrix = np.column_stack((rows, cols, values))
    return sparse_matrix

def normalize(mat, is_log=False, is_min_max=False):
    if is_log:
        mat = np.log1p(mat)

    if is_min_max:
        min_val = np.min(mat)
        max_val = np.max(mat)
        denom = max_val - min_val
        if denom == 0:
            fill_value = 1.0 if min_val > 0 else 0.0
            mat = np.full_like(mat, fill_value, dtype=np.float32)
        else:
            mat = (mat - min_val) / denom

    return mat

def run_4dmax(timeframe: List[torch.Tensor], patch_size=64):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    np.random.seed(42)

    eta = 100
    alpha = 0.6
    lr = 0.0001
    epochs = 300

    start_t = 0
    end_t = 2
    step = 1
    taos = np.array([0, 1])
    ts = np.linspace(start_t, end_t, step)

    map_tao = {}
    row_tao = {}
    col_tao = {}
    hic_dist_tao = {}
    ifs_tao = {}
    n_tao = {}
    n_min_tao = {}

    sparse_matrix_size = patch_size * patch_size
    for key, val in enumerate(taos):
        dense_matrix = timeframe[val].squeeze().cpu().numpy()
        sparse_matrix = get_sparse_matrix(dense_matrix)

        # if sparse_matrix.shape[0] < (sparse_matrix_size - sparse_matrix_size*0.10):
        #     print(
        #         f"[Warning] Skipping due to 10% less sparse matrix")
        #     return np.nan

        map_tao[val] = sparse_matrix
        row_tao[val] = (map_tao[val][:, 0].astype(int)).astype(int)
        col_tao[val] = (map_tao[val][:, 1].astype(int)).astype(int)
        ifs_tao[val] = map_tao[val][:, 2]
        hic_dist_tao[val] = ut.if2dist(ifs_tao[val], alpha)
        n_tao[val] = np.max((row_tao[val], col_tao[val]))
        n_min_tao[val] = np.min((row_tao[val], col_tao[val]))

    n_max = n_tao[list(n_tao.keys())[0]]
    n_max = np.max(list(n_tao.values()))
    n_min = np.min(list(n_min_tao.values()))

    for key, val in enumerate(taos):
        row_tao[val] = row_tao[val] - n_min_tao[val]
        col_tao[val] = col_tao[val] - n_min_tao[val]
    struc_t = np.random.rand(ts.shape[0], n_max+1-n_min, 3)

    GPU = True
    if GPU:
        for i in hic_dist_tao.keys():
            hic_dist_tao[i] = cp.array(hic_dist_tao[i])
        struc_t = cp.array(struc_t)
        ts = cp.array(ts)
        taos = cp.array(taos)

        for e in range(0, epochs):
            likelihood_loss = li.likelihoodlossGPU(
                hic_dist_tao, row_tao, col_tao, struc_t, ts, taos, n_max, n_min)
            movement_loss = mv.movementLossGPU(struc_t)
            struc_t -= lr*(likelihood_loss+(eta*movement_loss))

    else:
        for e in range(0, epochs):
            likelihood_loss = li.likelihoodloss(
                hic_dist_tao, row_tao, col_tao, struc_t, ts, taos, n_max, n_min)
            movement_loss = mv.movementLoss(struc_t)
            struc_t -= lr*(likelihood_loss+(eta*movement_loss))

    pred = ut.loadStrucAtTimeAsMat(struc_t, 0)
    pr, pc = pred.shape
    if pr != pc:
        print(
            f"[Warning] Skipping due to non-square predicted matrix")
        count += 1
        return np.nan

    pad_amount = patch_size - pr
    if pad_amount > 0:
        print(
            f"[Warning] Skipping due to shape mismatch between predicted {pred.shape} and ground truth {timeframe[0].shape}")
        # pred = np.pad(pred, ((0, pad_amount), (0, pad_amount)),
        #               mode='constant', constant_values=0)
        return np.nan

    pred_tensor = torch.tensor(pred).unsqueeze(
        0).unsqueeze(0).float().to('cuda')

    return pred_tensor


# with open(CSV_FILENAME, 'w') as f:
#     f.write(','.join(columns) + '\n')
#     for resolution in RESOLUTIONS:
#         for patch in PATCHES:
#             for organism in DATASET.keys():
#                 for sample in DATASET[organism].keys():
#                     for condition in DATASET[organism][sample].keys():
#                         triplet_list = DATASET[organism][sample][condition]['triplets']
#                         for triplet in triplet_list:
#                             timeframe_name = "_".join(
#                                 name.split('_')[-1] for name in triplet)
#                             for chromosome in CHROMOSOMES[organism]:
#                                 dict_filename = f"{DICT_DIR}/test_{resolution}_{patch}_{organism}_{triplet[1]}_{chromosome}.txt"
#                                 print(f"Running 4DMax for {dict_filename}...")
#                                 plot_filename = f"{OUTPUT_DIR}/plot_{resolution}_{patch}_{organism}_{sample}_{triplet[1]}_{chromosome}.png"
#                                 psnr, ssim, scc, genomedisco, hicrep, lpips = run_4dmax(
#                                     dict_filename, DATA_DIR, patch_size=patch, resolution=resolution, plot_filename=plot_filename)
#                                 print(
#                                     f"Results for {dict_filename}: PSNR={psnr}, SSIM={ssim}, SCC={scc}, GenomeDISCO={genomedisco}, HiCRep={hicrep}, LPIPS={lpips}")

#                                 row = {
#                                     "Resolution": resolution,
#                                     "Patch Size": patch,
#                                     "organism": organism,
#                                     "sample": sample,
#                                     "condition": condition,
#                                     "Timeframe": triplet[1],
#                                     "chromosome": chromosome,
#                                     "PSNR": psnr,
#                                     "SSIM": ssim,
#                                     "SCC": scc,
#                                     "GenomeDISCO": genomedisco,
#                                     "HiCRep": hicrep,
#                                     "LPIPS": lpips
#                                 }
#                                 print(
#                                     f"Writing results for {dict_filename}...")
#                                 f.write(','.join(str(value)
#                                         for value in row.values()) + '\n')
