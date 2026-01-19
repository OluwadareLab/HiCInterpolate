import os
from unittest import result
import numpy as np
import math
import cooler as cool
from scipy.ndimage import gaussian_filter as sp_gf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as sp_gf


ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data"
RESOLUTIONS = [10000]
BALANCE_COOL = True
PATCHES = [64]
ORGANISMS = ["human"]
SAMPLES = [
    [
        "dmso_control",
        "hela_s3_r1"
    ]
]
SUBSAMPLES = [
    [
        [
            "control"
        ],
        [
            "r1"
        ],
    ]
]
FILENAME_LIST = [
    [
        [
            "4DNFIP9EJSOM_dmso_0m",
            "4DNFI7T93SHL_dmso_30m",
            "4DNFICF2Z2TG_dmso_60m"
        ],
        [
            "4DNFIJMS2ODT_hela_s3_r1_30m",
            "4DNFI49F3LJ4_hela_s3_r1_60m",
            "4DNFI65MQOIJ_hela_s3_r1_90m"
        ]
    ]
]
CHROMOSOMES = [11, 13, 15, 17, 19, 21]

_CMAP = "Reds"
_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99


def plot_hic_map(matrix, filename):
    plt.imshow(matrix, cmap=_CMAP)
    plt.title(f"{filename}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300, format='pdf')
    plt.savefig(f"{filename}.png", dpi=300, format='png')
    plt.close()


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch]
    submatrix = submatrix.astype(np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)


def get_padded_matrix(_matrix, patch_size):
    h, w = _matrix.shape
    _extension = int(math.ceil(h/patch_size)*patch_size)
    padded = np.full(shape=(_extension, _extension),
                     fill_value=_EPSILON, dtype=np.float32)
    padded[:_matrix.shape[0], :_matrix.shape[1]] = _matrix

    return padded


def generate_patch(mat_0, mat_y, max_contact_frequency, mat_1, organism, sample, resolution, chromosome, sub_sample, timeframe_name, counter, output_root_path):
    for patch, i in zip(PATCHES, range(0, len(counter))):
        desired_path = f"{organism}/{sample}/{sub_sample}/{timeframe_name}/{str(resolution)}/{chromosome}"

        info_file = f"{output_root_path}/{patch}/{desired_path}/info.txt"
        os.makedirs(os.path.dirname(info_file), exist_ok=True)
        with open(info_file, "a") as file:
            file.write(f"organism: {organism}\n")
            file.write(f"sample: {sample}\n")
            file.write(f"sub_sample: {sub_sample}\n")
            file.write(f"timeframe: {timeframe_name}\n")
            file.write(f"resolution: {resolution}\n")
            file.write(f"chromosome: {chromosome}\n")
            file.write(f"bins: {mat_y.shape[0]}\n")
            file.write(f"max_contact_frequency: {max_contact_frequency}\n")
            file.close()

        mat_y = get_padded_matrix(mat_y, patch)
        mat_0 = get_padded_matrix(mat_0, patch)
        mat_1 = get_padded_matrix(mat_1, patch)

        ds_file = f"{output_root_path}/{patch}/{desired_path}/dataset_dict.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {timeframe_name} > {resolution} > chr{chromosome}")
            row, col = mat_y.shape
            bin_inc = patch

            r = 0
            while (r+patch <= row):
                c = 0
                while (c+patch <= col):
                    folder = f"{counter[i]:08d}"
                    path = f"{desired_path}/{folder}"
                    file.write(path+"\n")
                    path = f"{output_root_path}/{patch}/{path}"
                    os.makedirs(path, exist_ok=True)
                    save_img(mat_0, r, c, patch, path, "img1")
                    save_img(mat_y, r, c, patch, path, "img2")
                    save_img(mat_1, r, c, patch, path, "img3")
                    counter[i] += 1
                    c += bin_inc
                r += bin_inc
        print(f"Total patches generated: {counter[i]-1}")
        counter[i] = 1
    return counter


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


def generate_ds(organisms, samples, subsamples, filename_list, output_root_path: str, gf: bool, log: bool, clip: bool):
    for organism, org_samples, org_subsamples, org_filenames in zip(organisms, samples, subsamples, filename_list):
        for sub_sample, sample_filenames in zip(org_subsamples, org_filenames):
            for resolution in RESOLUTIONS:
                cool_0 = cool.Cooler(
                    f"{ROOT_PATH}/time_series_data/{organism}/sample/{org_samples[0]}/{sample_filenames[0]}_{resolution}_KR.cool")
                cool_y = cool.Cooler(
                    f"{ROOT_PATH}/time_series_data/{organism}/sample/{org_samples[0]}/{sample_filenames[1]}_{resolution}_KR.cool")
                cool_1 = cool.Cooler(
                    f"{ROOT_PATH}/time_series_data/{organism}/sample/{org_samples[0]}/{sample_filenames[2]}_{resolution}_KR.cool")
                timeframe_name = "_".join(name.split(
                    '_')[-1] for name in sample_filenames[:3])
                for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                    fetch = f"{chromosome}:{0}-{chr_size}"
                    chr_mat_0 = cool_0.matrix(
                        balance=BALANCE_COOL).fetch(fetch)
                    chr_mat_0 = get_norm_mat(
                        matrix=chr_mat_0, gf=gf, log=log, clip=clip)
                    chr_mat_y = cool_y.matrix(
                        balance=BALANCE_COOL).fetch(fetch)
                    desired_path = f"{organism}/{org_samples[0]}/{sub_sample[0]}/{timeframe_name}/{str(resolution)}/{chromosome}"
                    os.makedirs(
                        f"{output_root_path}/{PATCHES[0]}/{desired_path}", exist_ok=True)
                    np.save(
                        f"{output_root_path}/{PATCHES[0]}/{desired_path}/y.npy", chr_mat_y)
                    max_contact_frequency = np.max(chr_mat_y)
                    chr_mat_y = get_norm_mat(
                        matrix=chr_mat_y, gf=gf, log=log, clip=clip)
                    chr_mat_1 = cool_1.matrix(
                        balance=BALANCE_COOL).fetch(fetch)
                    chr_mat_1 = get_norm_mat(
                        matrix=chr_mat_1, gf=gf, log=log, clip=clip)
                    counter = [1]
                    counter = generate_patch(mat_0=chr_mat_0, mat_y=chr_mat_y, max_contact_frequency=max_contact_frequency, mat_1=chr_mat_1,
                                             organism=organism, sample=org_samples[0], resolution=resolution, chromosome=chromosome, sub_sample=sub_sample[0], timeframe_name=timeframe_name, counter=counter, output_root_path=output_root_path)


if __name__ == "__main__":
    try:
        output_root_path = f"{ROOT_PATH}/inference/kr_gf"
        generate_ds(organisms=ORGANISMS, samples=SAMPLES, subsamples=SUBSAMPLES,
                    filename_list=FILENAME_LIST, output_root_path=output_root_path, gf=True, log=False, clip=False)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
