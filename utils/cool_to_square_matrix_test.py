import os
import numpy as np
import cooler as cool

INPUT_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets"
OUTPUT_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/test_triplets/mm_triplets"

RESOLUTIONS = [25000, 10000, 5000]
BALANCE_COOL = False
PATCHES = [64, 128]
CLIPPING_PERCENTILE = 99.99
PATCH_OVERLAP_RATIO = 0.2

COUNTER = {
    5000: {64: 0, 128: 0},
    10000: {64: 0, 128: 0},
    25000: {64: 0, 128: 0}
}

CHROMOSOMES = ["11", "12", "13", "14", "15", "16",
               "17", "18", "19", "20", "21", "22"]

TEST_DATASET = {
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
        },
        "hct116": {
            "1": {
                "triplets":
                [
                    ["4DNFIDBFENL7_hct116_1_20m",
                     "4DNFI9ZUXG61_hct116_1_40m",
                     "4DNFIAUMRM2S_hct116_1_60m"]
                ]
            },
            "2": {
                "triplets":
                [
                    ["4DNFIAAH19VM_hct116_2_20m",
                     "4DNFI7QUSU5J_hct116_2_40m",
                     "4DNFIXEB4UZO_hct116_2_60m"]
                ]
            },
        },
        "hela_s3": {
            "r2": {
                "triplets":
                [
                    ["4DNFIX6ZXCA8_hela_s3_r2_30m",
                     "4DNFIEVR81FS_hela_s3_r2_60m",
                     "4DNFIAUI6BBI_hela_s3_r2_90m"]
                ]
            },
            "r3": {
                "triplets":
                [
                    ["4DNFICFZGFAV_hela_s3_r3_30m",
                     "4DNFIQXCZVVA_hela_s3_r3_60m",
                     "4DNFIB6PJFJ3_hela_s3_r3_90m"]
                ]
            }
        }
    },
    "mouse": {
        "embryo": {
            "development": {
                "triplets": [
                    ["4DNFIN8F14CS_sperm",
                     "4DNFIVCJKHMN_mii_oocyte",
                     "4DNFI1EYIGOC_zygote"],

                    ["4DNFIVCJKHMN_mii_oocyte",
                     "4DNFI1EYIGOC_zygote",
                     "4DNFIK4CECUH_early2_cell"],

                    ["4DNFI1EYIGOC_zygote",
                     "4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell"],

                    ["4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell"],

                    ["4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm"],

                    ["4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm",
                     "4DNFI5IAH9H1_mes_cell"]
                ]
            }
        }
    }
}


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch]
    submatrix = submatrix.astype(np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)


def normalize(mat, is_log=False, is_min_max=False):
    if is_log:
        mat = np.log1p(mat)
    max_val = None
    if is_min_max:
        min_val = np.min(mat)
        max_val = np.max(mat)
        if max_val - min_val > 0:
            mat = (mat - min_val) / (max_val - min_val)
    return mat, max_val


def pad_matrix_for_patches_numpy(matrix, patch_size=64):
    h, w = matrix.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    pad_width = [(0, 0)] * (matrix.ndim - 2)
    pad_width.append((0, pad_h))
    pad_width.append((0, pad_w))
    return np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)


def generate_patch(mat_0, mat_y, mat_1, organism, sample, subsample, resolution, chromosome, output_root_path, uuid, ds_dict_file):
    with open(ds_dict_file, "a") as dict_file:
        for patch in PATCHES:
            print(
                f"[INFO] generating patches({patch}X{patch}) for {resolution} > {organism} > {sample} > {subsample} > {uuid} > chr{chromosome}")
            mat_0 = pad_matrix_for_patches_numpy(mat_0, patch)
            mat_y = pad_matrix_for_patches_numpy(mat_y, patch)
            mat_1 = pad_matrix_for_patches_numpy(mat_1, patch)
            t_row, t_col = mat_y.shape

            for row in range(0, t_row, patch):
                for col in range(0, t_col+1, patch):
                    folder = f"{COUNTER[resolution][patch]:05d}"
                    record = f"{str(resolution)}/{str(patch)}/{organism}/{sample}/{subsample}/{str(uuid)}/{chromosome}/{folder}"
                    dict_file.write(record + "\n")
                    image_path = f"{OUTPUT_PATH}/{record}"
                    os.makedirs(image_path, exist_ok=True)
                    save_img(mat_0, row, col, patch, image_path, "img1")
                    save_img(mat_y, row, col, patch, image_path, "img2")
                    save_img(mat_1, row, col, patch, image_path, "img3")
                    COUNTER[resolution][patch] += 1

        dict_file.flush()
        dict_file.close()


def prepare_triplates():
    for resolution in RESOLUTIONS:
        ds_dict_file = f"{OUTPUT_PATH}/dataset_dict_{resolution}.txt"
        os.makedirs(os.path.dirname(ds_dict_file), exist_ok=True)

        for organism, samples in TEST_DATASET.items():
            for sample, subsamples in samples.items():
                for subsample, content in subsamples.items():
                    for triplet in content["triplets"]:
                        filepath0 = f"{INPUT_PATH}/{organism}/{sample}/{subsample}/{triplet[0]}_{resolution}_KR.cool"
                        filepath1 = f"{INPUT_PATH}/{organism}/{sample}/{subsample}/{triplet[1]}_{resolution}_KR.cool"
                        filepath2 = f"{INPUT_PATH}/{organism}/{sample}/{subsample}/{triplet[2]}_{resolution}_KR.cool"

                        cool_0 = cool.Cooler(filepath0)
                        cool_y = cool.Cooler(filepath1)
                        cool_1 = cool.Cooler(filepath2)
                        uuid = triplet[1]
                        for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                            fetch = f"{chromosome}:{0}-{chr_size}"
                            mat_0 = cool_0.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            mat_y = cool_y.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            mat_1 = cool_1.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            mat_0, max_val_0 = normalize(
                                mat_0, is_log=False, is_min_max=True)
                            mat_y, max_val_y = normalize(
                                mat_y, is_log=False, is_min_max=True)
                            mat_1, max_val_1 = normalize(
                                mat_1, is_log=False, is_min_max=True)
                            generate_patch(mat_0=mat_0, mat_y=mat_y, mat_1=mat_1, resolution=resolution,
                                           organism=organism, sample=sample, subsample=subsample, uuid=uuid, chromosome=chromosome, output_root_path=OUTPUT_PATH, ds_dict_file=ds_dict_file)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
