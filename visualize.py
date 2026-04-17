from jinja2 import Environment
import os
import numpy as np
import cooler as cool

from norm_visualize import COUTER

ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets"
OUTPUT_ROOT_PATH = f"{ROOT_PATH}/triplates/raw/"
_EPSILON = 1e-8
RESOLUTIONS = [25000, 10000, 5000]
BALANCE_COOL = False
PATCHES = [64, 128, 256]
PATCH_OVERLAP_RATIO = 0.2

COUNTER = {
    5000: {64: 0, 128: 0, 256: 0, 512: 0},
    10000: {64: 0, 128: 0, 256: 0, 512: 0},
    25000: {64: 0, 128: 0, 256: 0, 512: 0}
}


DATA = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                [
                    [
                        "4DNFIP9EJSOM_dmso_control_0m",
                        "4DNFI7T93SHL_dmso_control_30m",
                        "4DNFICF2Z2TG_dmso_control_60m",
                    ],
                    [
                        "4DNFI7T93SHL_dmso_control_30m",
                        "4DNFICF2Z2TG_dmso_control_60m",
                        "4DNFILL624WG_dmso_control_90m",
                    ],
                    [
                        "4DNFICF2Z2TG_dmso_control_60m",
                        "4DNFILL624WG_dmso_control_90m",
                        "4DNFIC4GB8UM_dmso_control_120m",
                    ],
                ]
            }
        },

        "dtag": {
            "v1": {
                "triplets":
                [
                    ["4DNFI5EAPQTI_dtag_v1_0m",
                     "4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m"],

                    ["4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m"],

                    ["4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m",
                     "4DNFIPZCCTV6_dtag_v1_120m"]
                ]
            }
        },

        "hct116": {
            "1": {
                "triplets":
                [
                    ["4DNFIDBFENL7_hct116_1_20m",
                     "4DNFI9ZUXG61_hct116_1_40m",
                     "4DNFIAUMRM2S_hct116_1_60m"
                     ],

                    ["4DNFIV56OFE3_hct116_1_auxin_20m",
                     "4DNFIBCIA62Q_hct116_1_auxin_40m",
                     "4DNFIQRTP7NM_hct116_1_auxin_60m"]
                ]
            },
            "2": {
                "triplets":
                [
                    ["4DNFIAAH19VM_hct116_2_20m",
                     "4DNFI7QUSU5J_hct116_2_40m",
                     "4DNFIXEB4UZO_hct116_2_60m"],

                    ["4DNFI5IZNXIO_hct116_2_no_transcription_360m_20m",
                     "4DNFIZK7W8GZ_hct116_2_no_transcription_360m_40m",
                     "4DNFISRP84FE_hct116_2_no_transcription_360m_60m"],

                    ["4DNFII16KXA7_hct116_2_no_transcription_60m_20m",
                     "4DNFIMIMLMD3_hct116_2_no_transcription_60m_40m",
                     "4DNFI2LY7B73_hct116_2_no_transcription_60m_60m"],

                    ["4DNFITUPI4HA_hct116_2_no_atp_120m_20m",
                     "4DNFIM7Q2FQQ_hct116_2_no_atp_120m_40m",
                     "4DNFISATK9PF_hct116_2_no_atp_120m_60m"],

                    ["4DNFIVC8OQPG_hct116_2_no_atp_30m_20m",
                     "4DNFI44JLUSL_hct116_2_no_atp_30m_40m",
                     "4DNFIBED48O1_hct116_2_no_atp_30m_60m"],

                    ["4DNFIDD9IF9T_hct116_2_no_replication_20m",
                     "4DNFIQWWATGK_hct116_2_no_replication_40m",
                     "4DNFI3NTD7B3_hct116_2_no_replication_60m"]
                ]
            },
        },

        "hela_s3": {
            "r1": {
                "triplets":
                [
                    ["4DNFIZZ77KD2_hela_s3_r1_30m",
                     "4DNFIOLO226X_hela_s3_r1_60m",
                     "4DNFIJMS2ODT_hela_s3_r1_90m"],

                    ["4DNFIJMS2ODT_hela_s3_r1_90m",
                     "4DNFI49F3LJ4_hela_s3_r1_105m",
                     "4DNFI65MQOIJ_hela_s3_r1_120m"],

                    ["4DNFI49F3LJ4_hela_s3_r1_105m",
                     "4DNFI65MQOIJ_hela_s3_r1_120m",
                     "4DNFIM4KEPRD_hela_s3_r1_135m"],

                    ["4DNFI65MQOIJ_hela_s3_r1_120m",
                     "4DNFIM4KEPRD_hela_s3_r1_135m",
                     "4DNFIIXBIZFC_hela_s3_r1_150m"],

                    ["4DNFIM4KEPRD_hela_s3_r1_135m",
                     "4DNFIIXBIZFC_hela_s3_r1_150m",
                     "4DNFIWDOOBVE_hela_s3_r1_165m"],

                    ["4DNFIIXBIZFC_hela_s3_r1_150m",
                     "4DNFIWDOOBVE_hela_s3_r1_165m",
                     "4DNFIDT9EB5M_hela_s3_r1_180m"],

                    ["4DNFIWDOOBVE_hela_s3_r1_165m",
                     "4DNFIDT9EB5M_hela_s3_r1_180m",
                     "4DNFIX2VUNV8_hela_s3_r1_195m"],

                    ["4DNFIDT9EB5M_hela_s3_r1_180m",
                     "4DNFIX2VUNV8_hela_s3_r1_195m",
                     "4DNFIEQHTV1R_hela_s3_r1_210m"],

                    ["4DNFIEQHTV1R_hela_s3_r1_210m",
                     "4DNFIFW7GA64_hela_s3_r1_240m",
                     "4DNFIXGXD67I_hela_s3_r1_270m"],

                    ["4DNFIFW7GA64_hela_s3_r1_240m",
                     "4DNFIXGXD67I_hela_s3_r1_270m",
                     "4DNFIA7GB1NB_hela_s3_r1_300m"]
                ]
            },
            "r2": {
                "triplets":
                [
                    ["4DNFIX6ZXCA8_hela_s3_r2_30m",
                     "4DNFIEVR81FS_hela_s3_r2_60m",
                     "4DNFIAUI6BBI_hela_s3_r2_90m"],

                    ["4DNFIEVR81FS_hela_s3_r2_60m",
                     "4DNFIAUI6BBI_hela_s3_r2_90m",
                     "4DNFIAFEE9G2_hela_s3_r2_120m"],

                    ["4DNFIAUI6BBI_hela_s3_r2_90m",
                     "4DNFIAFEE9G2_hela_s3_r2_120m",
                     "4DNFIPZBEXCP_hela_s3_r2_150m"],

                    ["4DNFIAFEE9G2_hela_s3_r2_120m",
                     "4DNFIPZBEXCP_hela_s3_r2_150m",
                     "4DNFIWPKRZGU_hela_s3_r2_180m"],

                    ["4DNFIPZBEXCP_hela_s3_r2_150m",
                     "4DNFIWPKRZGU_hela_s3_r2_180m",
                     "4DNFIMD9QNDX_hela_s3_r2_210m"],

                    ["4DNFIWPKRZGU_hela_s3_r2_180m",
                     "4DNFIMD9QNDX_hela_s3_r2_210m",
                     "4DNFIATA1HD5_hela_s3_r2_240m"],

                    ["4DNFIMD9QNDX_hela_s3_r2_210m",
                     "4DNFIATA1HD5_hela_s3_r2_240m",
                     "4DNFIH9U4I7I_hela_s3_r2_270m"],

                    ["4DNFIATA1HD5_hela_s3_r2_240m",
                     "4DNFIH9U4I7I_hela_s3_r2_270m",
                     "4DNFIZ95S6TR_hela_s3_r2_300m"]
                ]
            },
            "r3": {
                "triplets":
                [
                    ["4DNFICFZGFAV_hela_s3_r3_30m",
                     "4DNFIQXCZVVA_hela_s3_r3_60m",
                     "4DNFIB6PJFJ3_hela_s3_r3_90m"],

                    ["4DNFIB6PJFJ3_hela_s3_r3_90m",
                     "4DNFIX97731O_hela_s3_r3_105m",
                     "4DNFIYQYZOTO_hela_s3_r3_120m"],

                    ["4DNFIX97731O_hela_s3_r3_105m",
                     "4DNFIYQYZOTO_hela_s3_r3_120m",
                     "4DNFIPXU7V25_hela_s3_r3_135m"],

                    ["4DNFIYQYZOTO_hela_s3_r3_120m",
                     "4DNFIPXU7V25_hela_s3_r3_135m",
                     "4DNFIL39PR76_hela_s3_r3_150m"],

                    ["4DNFIPXU7V25_hela_s3_r3_135m",
                     "4DNFIL39PR76_hela_s3_r3_150m",
                        "4DNFIYLJ3R3B_hela_s3_r3_165m"],

                    ["4DNFIL39PR76_hela_s3_r3_150m",
                     "4DNFIYLJ3R3B_hela_s3_r3_165m",
                     "4DNFIL51WBN6_hela_s3_r3_180m"],

                    ["4DNFIYLJ3R3B_hela_s3_r3_165m",
                     "4DNFIL51WBN6_hela_s3_r3_180m",
                     "4DNFI6SFPUDA_hela_s3_r3_195m"],

                    ["4DNFIL51WBN6_hela_s3_r3_180m",
                     "4DNFI6SFPUDA_hela_s3_r3_195m",
                     "4DNFI2KM22QR_hela_s3_r3_210m"],

                    ["4DNFI2KM22QR_hela_s3_r3_210m",
                     "4DNFIVF8Q45U_hela_s3_r3_240m",
                     "4DNFI2RN3WFP_hela_s3_r3_270m"],

                    ["4DNFIVF8Q45U_hela_s3_r3_240m",
                     "4DNFI2RN3WFP_hela_s3_r3_270m",
                     "4DNFI4TJTL7A_hela_s3_r3_300m"]
                ]
            },
        }
    }
}


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch].astype(np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)


def generate_patch(mat_0, mat_y, mat_1, organism, sample, subsample, resolution, chromosome, output_root_path, uuid):
    for patch in PATCHES:
        patch_path = f"{output_root_path}/{patch}"
        os.makedirs(patch_path, exist_ok=True)

        ds_dict_file = f"{patch_path}/dataset_dict_{resolution}.txt"
        os.makedirs(os.path.dirname(ds_dict_file), exist_ok=True)

        with open(ds_dict_file, "a") as dict_file:
            print(
                f"[INFO] generating patches({patch}X{patch}) for {organism} > {sample} > {subsample} > {resolution} > chr{chromosome}")

            t_row, t_col = mat_y.shape
            bin_inc = int(patch*(1-PATCH_OVERLAP_RATIO))
            window = [0]
            for win in window:
                r = win
                c = 0
                while (r+patch <= t_row and c+patch <= t_col):
                    if r < 0 or c < 0:
                        c += bin_inc
                        r += bin_inc
                        continue
                    COUNTER[resolution][patch] += 1
                    folder = f"{COUTER[resolution][patch]:08d}"
                    record = f"{organism}/{sample}/{subsample}/{str(resolution)}/{chromosome}/{folder}"
                    dict_file.write(record + "\n")
                    image_path = f"{patch_path}/{record}"
                    os.makedirs(image_path, exist_ok=True)
                    save_img(mat_0, r, c, patch, image_path, "img1")
                    save_img(mat_y, r, c, patch, image_path, "img2")
                    save_img(mat_1, r, c, patch, image_path, "img3")
                    c += bin_inc
                    r += bin_inc


def nan_zero_to_eps(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    matrix[matrix == 0] = _EPSILON
    return matrix


def prepare_triplates():
    for resolution in RESOLUTIONS:
        for organism, samples in DATA.items():
            for sample, subsamples in samples.items():
                for subsample, content in subsamples.items():
                    for triplet in content["triplets"]:
                        print(
                            f"Processing {organism} > {sample} > {subsample} > {resolution} > {triplet}")

                        filepath0 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[0]}_{resolution}_KR.cool"
                        filepath1 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[1]}_{resolution}_KR.cool"
                        filepath2 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[2]}_{resolution}_KR.cool"

                        cool_0 = cool.Cooler(filepath0)
                        cool_y = cool.Cooler(filepath1)
                        cool_1 = cool.Cooler(filepath2)
                        uuid = triplet[0] + "_" + triplet[1] + "_" + triplet[2]
                        for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                            fetch = f"{chromosome}:{0}-{chr_size}"
                            mat_0 = nan_zero_to_eps(cool_0.matrix(
                                balance=BALANCE_COOL).fetch(fetch))
                            mat_y = nan_zero_to_eps(cool_y.matrix(
                                balance=BALANCE_COOL).fetch(fetch))
                            mat_1 = nan_zero_to_eps(cool_1.matrix(
                                balance=BALANCE_COOL).fetch(fetch))
                            generate_patch(mat_0=mat_0, mat_y=mat_y, mat_1=mat_1,
                                           organism=organism, sample=sample, subsample=subsample, resolution=resolution, chromosome=chromosome, output_root_path=OUTPUT_ROOT_PATH, uuid=uuid)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
