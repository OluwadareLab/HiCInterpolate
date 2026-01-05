import os
import numpy as np
import math
import cooler as cool
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as sp_gf


ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data"
RESOLUTIONS = [10000]
BALANCE_COOL = True
PATCHES = [64]

ORGANISMS = [
    "human"
]

SAMPLES = [
    [
        "dmso_control",
        "dtag_v1",
        "hela_s3_r1",
        "hct116_2"
    ]
]

SUBSAMPLES = [
    [
        [
            "control"
        ],
        [
            "v1"
        ],
        [
            "r1"
        ],
        [
            "noatp30m",
            "noatp120m",
            "notranscription60m",
            "notranscription360m"
        ]
    ]
]

FILENAME_LIST = [
    [
        [
            [
                "4DNFIP9EJSOM_dmso_0m",
                "4DNFI7T93SHL_dmso_30m",
                "4DNFICF2Z2TG_dmso_60m"
            ]
        ],
        [
            [
                "4DNFI5EAPQTI_dtag_v1_0m",
                "4DNFIY1TCVLX_dtag_v1_30m",
                "4DNFIXWT5U42_dtag_v1_60m"
            ]
        ],
        [
            [
                "4DNFIZZ77KD2_hela_s3_r1_30m",
                "4DNFIOLO226X_hela_s3_r1_60m",
                "4DNFIJMS2ODT_hela_s3_r1_90m"
            ]
        ],
        [
            [
                "4DNFIVC8OQPG_hct116_2_noatp30m_20m",
                "4DNFI44JLUSL_hct116_2_noatp30m_40m",
                "4DNFIBED48O1_hct116_2_noatp30m_60m"
            ],
            [
                "4DNFITUPI4HA_hct116_2_noatp120m_20m",
                "4DNFIM7Q2FQQ_hct116_2_noatp120m_40m",
                "4DNFISATK9PF_hct116_2_noatp120m_60m"
            ],
            [
                "4DNFII16KXA7_hct116_2_notranscription60m_20m",
                "4DNFIMIMLMD3_hct116_2_notranscription60m_40m",
                "4DNFI2LY7B73_hct116_2_notranscription60m_60m"
            ],
            [
                "4DNFI5IZNXIO_hct116_2_notranscription360m_20m",
                "4DNFIZK7W8GZ_hct116_2_notranscription360m_40m",
                "4DNFISRP84FE_hct116_2_notranscription360m_60m"
            ]
        ]
    ]
]

CHROMOSOMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']


_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99


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


def generate_patch(mat_0, mat_y, mat_1, patch, organism, sample, sub_sample, timeframe_name, resolution, chromosome, output_root_path):
    for patch in PATCHES:
        desired_path = f"{organism}/{sample}/{sub_sample}/{timeframe_name}/{str(resolution)}/{chromosome}"

        mat_y = get_padded_matrix(mat_y, patch)
        mat_0 = get_padded_matrix(mat_0, patch)
        mat_1 = get_padded_matrix(mat_1, patch)

        ds_file = f"{output_root_path}/{patch}/{desired_path}/dataset_dict.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {timeframe_name} > {resolution} > chr{chromosome}")

            row, col = mat_y.shape
            counter = 1
            r = 0
            while (r+patch <= row):
                c = 0
                while (c+patch <= col):
                    folder = f"{counter:08d}"
                    path = f"{desired_path}/{folder}"
                    file.write(path+"\n")
                    path = f"{output_root_path}/{patch}/{path}"
                    os.makedirs(path, exist_ok=True)

                    save_img(mat_0, r, c, patch, path, "img1")
                    save_img(mat_y, r, c, patch, path, "img2")
                    save_img(mat_1, r, c, patch, path, "img3")

                    counter += 1
                    c += patch
                r += patch

        print(f"Total patches generated: {counter-1}")


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


def write_info(matrix, patch, organism, sample, sub_sample, timeframe_name, resolution, chromosome, output_root_path):
    desired_path = f"{output_root_path}/{patch}/{organism}/{sample}/{sub_sample}/{timeframe_name}/{str(resolution)}/{chromosome}"
    os.makedirs(f"{desired_path}", exist_ok=True)
    np.save(f"{desired_path}/y.npy", matrix)

    info_file = f"{desired_path}/info.txt"
    os.makedirs(os.path.dirname(info_file), exist_ok=True)

    with open(info_file, "a") as file:
        file.write(f"organism\t{organism}\n")
        file.write(f"sample\t{sample}\n")
        file.write(f"sub_sample\t{sub_sample}\n")
        file.write(f"timeframe\t{timeframe_name}\n")
        file.write(f"resolution\t{resolution}\n")
        file.write(f"chromosome\t{chromosome}\n")
        file.write(f"bins: {matrix.shape[0]}\n")
        file.write(f"max_contact_frequency: {np.max(matrix)}\n")
        file.close()


def generate_ds(organisms, samples, subsamples, filename_list, output_root_path: str, gf: bool = False, log: bool = False, clip: bool = False):
    for patch in PATCHES:
        for organism, org_samples, org_subsamples, org_filenames in zip(organisms, samples, subsamples, filename_list):
            for sample, sam_sub_sample, sam_sample_filenames in zip(org_samples, org_subsamples, org_filenames):
                for sub_sample, sample_filenames in zip(sam_sub_sample, sam_sample_filenames):
                    for resolution in RESOLUTIONS:

                        # file_path = f"{ROOT_PATH}/time_series_data/{organism}/sample/{sample}/{sample_filenames[1]}_{resolution}_KR.cool"
                        # if os.path.exists(file_path):
                        #     print(f"{file_path} exists")
                        # else:
                        #     print(f"{file_path} does not exist")

                        cool_0 = cool.Cooler(
                            f"{ROOT_PATH}/time_series_data/{organism}/sample/{sample}/{sample_filenames[0]}_{resolution}_KR.cool")
                        cool_y = cool.Cooler(
                            f"{ROOT_PATH}/time_series_data/{organism}/sample/{sample}/{sample_filenames[1]}_{resolution}_KR.cool")
                        cool_1 = cool.Cooler(
                            f"{ROOT_PATH}/time_series_data/{organism}/sample/{sample}/{sample_filenames[2]}_{resolution}_KR.cool")

                        timeframe_name = "_".join(name.split(
                            '_')[-1] for name in sample_filenames[:3])

                        for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                            fetch = f"{chromosome}:{0}-{chr_size}"
                            chr_mat_0 = cool_0.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            chr_mat_0 = get_norm_mat(matrix=chr_mat_0, gf=gf)
                            chr_mat_y = cool_y.matrix(
                                balance=BALANCE_COOL).fetch(fetch)

                            write_info(chr_mat_y, patch, organism, sample, sub_sample,
                                       timeframe_name, resolution, chromosome, output_root_path)
                            chr_mat_y = get_norm_mat(matrix=chr_mat_y, gf=gf)

                            chr_mat_1 = cool_1.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            chr_mat_1 = get_norm_mat(matrix=chr_mat_1, gf=gf)

                            generate_patch(mat_0=chr_mat_0, mat_y=chr_mat_y, mat_1=chr_mat_1, patch=patch,
                                           organism=organism, sample=sample, sub_sample=sub_sample, timeframe_name=timeframe_name, resolution=resolution, chromosome=chromosome, output_root_path=output_root_path)


if __name__ == "__main__":
    try:
        output_root_path = f"{ROOT_PATH}/inference/kr_gf"
        generate_ds(organisms=ORGANISMS, samples=SAMPLES, subsamples=SUBSAMPLES,
                    filename_list=FILENAME_LIST, output_root_path=output_root_path, gf=True)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
