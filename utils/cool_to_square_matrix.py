import os
import numpy as np
import cooler as cool
from scipy.ndimage import gaussian_filter as sp_gf
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as cp_gf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/time_series_data"
RESOLUTIONS = [10000]
BALANCE_COOL = True
PATCHES = [64]
_CMAP = "Reds"
_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99
PATCH_OVERLAP_RATIO = 0.2

ORGANISMS = ["human"]
SAMPLES = [[
    "dmso_control",
    "dtag_v1",
    "hct116_1",
    "hct116_2",
    "hela_s3_r1"
]]

FILENAME_LIST = [[[["4DNFI7T93SHL_dmso_30m",
                   "4DNFICF2Z2TG_dmso_60m",
                    "4DNFILL624WG_dmso_90m"]],
                 [["4DNFIY1TCVLX_dtag_v1_30m",
                   "4DNFIXWT5U42_dtag_v1_60m",
                   "4DNFIHTFIMGG_dtag_v1_90m"]],
                 [["4DNFIDBFENL7_hct116_20m",
                   "4DNFI9ZUXG61_hct116_40m",
                   "4DNFIAUMRM2S_hct116_60m"]],
                 [["4DNFIAAH19VM_hct116_2_20m",
                   "4DNFI7QUSU5J_hct116_2_40m",
                   "4DNFIXEB4UZO_hct116_2_60m"]],
                 [["4DNFIJMS2ODT_hela_s3_r1_105m",
                   "4DNFI49F3LJ4_hela_s3_r1_120m",
                   "4DNFI65MQOIJ_hela_s3_r1_135m"],
                  ["4DNFIM4KEPRD_hela_s3_r1_150m",
                   "4DNFIIXBIZFC_hela_s3_r1_165m",
                     "4DNFIWDOOBVE_hela_s3_r1_180m"],
                  ["4DNFIDT9EB5M_hela_s3_r1_180m",
                     "4DNFIX2VUNV8_hela_s3_r1_195m",
                     "4DNFIEQHTV1R_hela_s3_r1_210m"]]
                  ]]


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


def generate_patch(mat_0, mat_y, mat_1, organism, sample, resolution, chromosome, sub_sample, counter, output_root_path):
    for patch, i in zip(PATCHES, range(0, len(counter))):
        ds_file = f"{output_root_path}/{patch}/dataset_dict.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {resolution} > chr{chromosome}")
            row, col = mat_y.shape
            bin_inc = int(patch*(1-PATCH_OVERLAP_RATIO))
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
                    path = f"{output_root_path}/{patch}/{path}"
                    os.makedirs(path, exist_ok=True)
                    save_img(mat_0, r, c, patch, path, "img1")
                    save_img(mat_y, r, c, patch, path, "img2")
                    save_img(mat_1, r, c, patch, path, "img3")
                    counter[i] += 1
                    c += bin_inc
                    r += bin_inc

    return counter


def get_cp_gf(matrix, sigma=0.75):
    try:
        with cp.cuda.Device(0):
            matrix_gpu = cp.asarray(matrix)
            result_gpu = cp_gf(matrix_gpu, sigma=sigma, mode='nearest')
            result_cpu = cp.asnumpy(result_gpu)
            del matrix_gpu, result_gpu

            cp._default_memory_pool.free_all_blocks()
            return result_cpu
    except cp.cuda.memory.OutOfMemoryError:
        print("[ERROR] CuPy ran out of GPU memory.")
        raise cp.cuda.memory.OutOfMemoryError


def raw_matrix(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    return matrix


def gf_norm(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    gf_matrix = sp_gf(matrix, 0.75)
    _min = np.min(gf_matrix)
    _max = np.max(gf_matrix)
    mm_matrix = (gf_matrix - _min)/(_max - _min)
    mm_matrix[mm_matrix == 0] = _EPSILON
    return mm_matrix


def min_max_norm(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    _min = np.min(matrix)
    _max = np.max(matrix)
    mm_matrix = (matrix - _min)/(_max - _min)
    mm_matrix[mm_matrix == 0] = _EPSILON
    return mm_matrix


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


def get_norm_mat(matrix, gf: bool = False, log: bool = False, clip: bool = False):
    mat = np.nan_to_num(matrix, nan=_EPSILON, posinf=_EPSILON, neginf=_EPSILON)
    if gf:
        mat = sp_gf(mat, 1.0)
    if log:
        mat = np.log1p(matrix)
    if clip:
        percentile_val = np.percentile(mat, CLIPPING_PERCENTILE)
        mat = np.clip(mat, _EPSILON, percentile_val)

    _min = np.min(mat)
    _max = np.max(mat)
    mat = (mat - _min)/(_max - _min)
    mat[mat == 0] = _EPSILON

    return mat


def generate_ds(organisms, samples, filename_list, output_root_path: str, gf: bool, log: bool, clip: bool):
    for organism, org_samples, org_filenames in zip(organisms, samples, filename_list):
        for sample, sample_filenames in zip(org_samples, org_filenames):
            for resolution in RESOLUTIONS:
                for filenames in sample_filenames:
                    cool_0 = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/sample/{sample}/{filenames[0]}_{resolution}_KR.cool")
                    cool_y = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/sample/{sample}/{filenames[1]}_{resolution}_KR.cool")
                    cool_1 = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/sample/{sample}/{filenames[2]}_{resolution}_KR.cool")

                    sub_sample = "_".join(name.split(
                        '_')[-1] for name in filenames[:3])
                    for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                        fetch = f"{chromosome}:{0}-{chr_size}"
                        chr_mat_0 = cool_0.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        chr_mat_0 = get_norm_mat(
                            matrix=chr_mat_0, gf=gf, log=log, clip=clip)
                        chr_mat_y = cool_y.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        chr_mat_y = get_norm_mat(
                            matrix=chr_mat_y, gf=gf, log=log, clip=clip)
                        chr_mat_1 = cool_1.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        chr_mat_1 = get_norm_mat(
                            matrix=chr_mat_1, gf=gf, log=log, clip=clip)
                        counter = [1]
                        counter = generate_patch(chr_mat_0, chr_mat_y, chr_mat_1,
                                                 organism, sample, resolution, chromosome, sub_sample, counter, output_root_path=output_root_path)


if __name__ == "__main__":
    try:
        output_root_path = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/upload"
        generate_ds(ORGANISMS, SAMPLES, FILENAME_LIST,
                    output_root_path=output_root_path, gf=True, log=False, clip=False)
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
