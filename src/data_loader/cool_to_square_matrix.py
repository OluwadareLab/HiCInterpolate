import os
import numpy as np
import cooler as cool
from scipy.ndimage import gaussian_filter
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data"
OUTPUT_ROOT_PATH = f"{ROOT_PATH}/triplets/kr_diag"
RESOLUTIONS = [10000]
BALANCE_COOL = True
PATCHES = [64, 128, 256, 512]
# _CMAP = "YlOrRd"
_CMAP = "Reds"
_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99
PATCH_OVERLAP_RATIO = 0.2

ORGANISMS = ["human"]
SAMPLES = [["atrial", "verticular", "hela_s3"]]
FILENAME_LIST = [[[["a_2d_4DNFIWUZLEWY",
                    "a_4d_4DNFI7PCOZ9I",
                    "a_6d_4DNFIWZYSI2D"]],
                  [["v_2d_4DNFI1P2EP7L",
                    "v_4d_4DNFI7C5YXNX",
                    "v_6d_4DNFI8I2WYXS"]],
                  [["0_5h_4DNFIZZ77KD2",
                    "1h_4DNFIOLO226X",
                    "1_5h_4DNFIJMS2ODT"],
                   ["1_75h_4DNFI49F3LJ4",
                    "2h_4DNFI65MQOIJ",
                    "2_25h_4DNFIM4KEPRD"],
                   ["2_5h_4DNFIIXBIZFC",
                    "2_75h_4DNFIWDOOBVE",
                    "3h_4DNFIDT9EB5M"],
                   ["3h_4DNFIDT9EB5M",
                    "3_25h_4DNFIX2VUNV8",
                    "3_5h_4DNFIEQHTV1R"],
                   ["3_5h_4DNFIEQHTV1R",
                    "4h_4DNFIFW7GA64",
                    "4_5h_4DNFIXGXD67I"],
                   ["5h_4DNFIA7GB1NB",
                    "6h_4DNFIVOJGWNP",
                    "7h_4DNFIW22BNB5"],
                   ["8h_4DNFIIFBC8WN",
                    "9h_4DNFI9ZBEBJH",
                    "10h_4DNFID4SLU53"],
                   ["10h_4DNFID4SLU53",
                    "11h_4DNFIODI1NUJ",
                    "12h_4DNFIJL26LFN"]]]]


def plot_hic_map(matrix, filename):
    plt.imshow(matrix, cmap=_CMAP)
    plt.title(f"{filename}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, format='png')
    plt.close()


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch]
    submatrix = submatrix.astype(np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)


def generate_patch(mat_0, mat_y, mat_1, organism, sample, resolution, chromosome, sub_sample, counter):
    for patch, i in zip(PATCHES, range(0, len(counter))):
        ds_file = f"{OUTPUT_ROOT_PATH}/{patch}/dataset_dict.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {resolution} > chr{chromosome}")
            row, col = mat_y.shape
            bin_inc = int(patch*(1-PATCH_OVERLAP_RATIO))
            # win_start = patch//2
            # window = [-win_start, 0, win_start]
            window = [0]
            for win in window:
                r = win
                c = 0
                while (r+patch <= row and c+patch <= col):
                    if r < 0 or c < 0:
                        c += bin_inc
                        r += bin_inc
                        continue
                    folder = f"{counter[i]:08d}"
                    path = f"{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}/{folder}"
                    file.write(path+"\n")
                    path = f"{OUTPUT_ROOT_PATH}/{patch}/{path}"
                    os.makedirs(path, exist_ok=True)
                    save_img(mat_0, r, c, patch, path, "img1")
                    save_img(mat_y, r, c, patch, path, "img2")
                    save_img(mat_1, r, c, patch, path, "img3")
                    counter[i] += 1
                    c += bin_inc
                    r += bin_inc

    return counter


def fast_gaussian_filter(matrix, sigma=4):
    try:
        with cp.cuda.Device(0):
            matrix_gpu = cp.asarray(matrix)
            result_gpu = gaussian_filter(
                matrix_gpu, sigma=sigma, mode='reflect')
            result_cpu = cp.asnumpy(result_gpu)
            del matrix_gpu, result_gpu
            cp._default_memory_pool.free_all_blocks()
            return result_cpu
    except cp.cuda.memory.OutOfMemoryError:
        print("[ERROR] CuPy ran out of GPU memory.")
        raise cp.cuda.memory.OutOfMemoryError


def draw_hic_map(x0: np.ndarray, y: np.ndarray, x1: np.ndarray, filename):
    data_groups = [x0, y, x1]
    titles = ["$x_0$", "$y_{t=0.5}$", "$x_1$"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = np.atleast_2d(axes)
    for i in range(len(data_groups)):
        ax = axes[0, i]
        matrix = data_groups[i]
        _min = np.min(matrix)
        _max = np.max(matrix)
        im = ax.imshow(matrix, cmap=_CMAP, vmin=_min, vmax=_max)
        ax.set_title(titles[i])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, format='png')
    plt.close()


def raw_matrix(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    return matrix


def min_max_norm(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)

    min_val = np.min(matrix)
    max_val = np.max(matrix)
    # min_max_matrix = (matrix - min_val) / (max_val - min_val + _EPSILON)
    min_max_matrix = matrix/max_val
    return min_max_matrix


def log_clip(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    log_matrix = np.log1p(matrix)
    percentile_val = np.percentile(log_matrix, CLIPPING_PERCENTILE)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)

    return clip_matrix


def rev_log_clip_min_max(matrix):
    mat = np.expm1(matrix)
    log_matrix = np.log1p(mat)
    return log_matrix


def log_clip_min_max(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    log_matrix = np.log1p(matrix)
    percentile_val = np.percentile(log_matrix, CLIPPING_PERCENTILE)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)
    norm_matrix = clip_matrix / percentile_val

    return norm_matrix


def normalization(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    log_matrix = np.log1p(matrix)
    percentile_val = np.percentile(log_matrix, CLIPPING_PERCENTILE)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)
    norm_matrix = clip_matrix / percentile_val

    return norm_matrix


def draw_patch():
    chr_mat_0 = np.load(
        "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCPlus/data/GM12878_replicate_down16_chr17_17.npy")
    submatrix = chr_mat_0[159][0]
    submatrix = submatrix.astype(np.float32)

    raw_mat = raw_matrix(submatrix)
    plot_hic_map(raw_mat, f"hicplus_raw_mat")

    min_max_mat = min_max_norm(submatrix)
    plot_hic_map(min_max_mat, f"hicplus_min_max_mat")

    log_clip_mat = log_clip(submatrix)
    plot_hic_map(log_clip_mat, f"hicplus_log_clip_mat")

    log_clip_mat_af_mm = rev_log_clip_min_max(log_clip_mat)
    plot_hic_map(log_clip_mat_af_mm,
                 f"hicplus_log_clip_mat_af_mm")

    log_clip_min_max_mat = log_clip_min_max(submatrix)
    plot_hic_map(log_clip_min_max_mat,
                 f"hicplus_log_clip_min_max_mat")


def generate_ds(organisms, samples, filename_list):
    for organism, org_samples, org_filenames in zip(organisms, samples, filename_list):
        for sample, sample_filenames in zip(org_samples, org_filenames):
            for resolution in RESOLUTIONS:
                for filenames in sample_filenames:
                    cool_0 = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/{sample}/{filenames[0]}_{resolution}_KR.cool")
                    cool_y = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/{sample}/{filenames[1]}_{resolution}_KR.cool")
                    cool_1 = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/{sample}/{filenames[2]}_{resolution}_KR.cool")

                    sub_sample = "_".join(name.split(
                        '_')[-1] for name in filenames[:3])
                    for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                        fetch = f"{chromosome}:{0}-{chr_size}"
                        chr_mat_0 = cool_0.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        chr_mat_0 = log_clip(chr_mat_0)
                        chr_mat_y = cool_y.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        chr_mat_y = log_clip(chr_mat_y)
                        chr_mat_1 = cool_1.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        chr_mat_1 = log_clip(chr_mat_1)
                        counter = [1, 1, 1, 1]
                        counter = generate_patch(chr_mat_0, chr_mat_y, chr_mat_1,
                                                 organism, sample, resolution, chromosome, sub_sample, counter)


if __name__ == "__main__":
    try:
        generate_ds(ORGANISMS, SAMPLES, FILENAME_LIST)
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
