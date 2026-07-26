import math
import os
import numpy as np
import cooler
import matplotlib.pyplot as plt
from random import seed

INPUT_DIR = '/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/hic'
RESOLUTIONS = [25000]
PATCHES = [64]
MODEL_BATCHES = [20]

CHROMOSOMES = {
    "human": ["10", "11", "15", "16", "20", "21"],
    "mouse": ["10", "15", "19"]
}

CHROMOSOME_SIZES = {
    "human": {
        "10": 133797422,
        "11": 135086622,
        "15": 101991189,
        "16": 90338345,
        "20": 64444167,
        "21": 46709983
    },
    "mouse": {
        "10": 130694993,
        "15": 104043685,
        "19": 61431566
    }
}

TEST_DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                    [
                        ["dmso_control_30m",
                         "dmso_control_60m",
                         "dmso_control_90m"]
                    ]
            }
        },
        "dtag": {
            "v1": {
                "triplets":
                    [
                        ["dtag_v1_30m",
                         "dtag_v1_60m",
                         "dtag_v1_90m"]
                    ]
            }
        }
    },
    "mouse": {
        "cerebellar_granule_neuron": {
            "control": {
                "triplets":
                [
                    ["cerebellar_granule_neuron_control_10080m",
                     "cerebellar_granule_neuron_control_11520m",
                     "cerebellar_granule_neuron_control_12960m"]
                ]
            }
        },
        "embryo": {
            "development": {
                "triplets": [
                    ["zygote",
                     "early2_cell",
                     "late2_cell"],

                    ["early2_cell",
                     "late2_cell",
                     "8cell"],

                    ["late2_cell",
                     "8cell",
                     "icm"]
                ]
            }
        }
    }
}

seed(42)


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


def save_patch(matrix1, matrix2, matrix3,  output_dir, out_sub_dir):
    np.save(os.path.join(output_dir,
                         output_dir,  'img1.npy'), matrix1)
    np.save(os.path.join(output_dir,
                         output_dir,  'img2.npy'), matrix2)
    np.save(os.path.join(output_dir,
                         output_dir,  'img3.npy'), matrix3)


def generate_triplets(output_dir, is_log=False, is_min_max=False):
    for resolution in RESOLUTIONS:
        for organism, samples in TEST_DATASET.items():
            for sample, conditions in samples.items():
                for condition, dataset in conditions.items():
                    for triplet in dataset["triplets"]:
                        uuid = triplet[0].split(
                            "_")[-1] + "_" + triplet[1].split("_")[-1] + "_" + triplet[2].split("_")[-1]

                        filename1 = f"{INPUT_DIR}/{triplet[0]}_{resolution}_KR.cool"
                        filename2 = f"{INPUT_DIR}/{triplet[1]}_{resolution}_KR.cool"
                        filename3 = f"{INPUT_DIR}/{triplet[2]}_{resolution}_KR.cool"

                        if not (os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3)):
                            print(
                                f"[WARNING] Missing files for {organism} > {sample} > {condition} > {uuid} > {resolution}")
                            continue

                        cool1 = cooler.Cooler(
                            f"{INPUT_DIR}/{triplet[0]}_{resolution}_KR.cool")
                        cool2 = cooler.Cooler(
                            f"{INPUT_DIR}/{triplet[1]}_{resolution}_KR.cool")
                        cool3 = cooler.Cooler(
                            f"{INPUT_DIR}/{triplet[2]}_{resolution}_KR.cool")

                        for chrom in cool2.chromnames:
                            if chrom not in CHROMOSOMES[organism]:
                                continue
                            print(
                                f"[INFO] Processing {organism} > {sample} > {condition} > {uuid} > {resolution} > chr{chrom}")

                            out_sub_dir = f'{resolution}/{organism}/{sample}/{condition}/{triplet[1]}/chr{chrom}'

                            matrix1 = normalize(cool1.matrix(balance=False).fetch(
                                chrom), is_log=is_log, is_min_max=is_min_max)
                            matrix2 = normalize(cool2.matrix(balance=False).fetch(
                                chrom), is_log=is_log, is_min_max=is_min_max)
                            matrix3 = normalize(cool3.matrix(balance=False).fetch(
                                chrom), is_log=is_log, is_min_max=is_min_max)

                            save_patch(
                                matrix1, matrix2, matrix3, cool2.chromsizes[chrom], output_dir, out_sub_dir)


if __name__ == "__main__":
    try:
        output_dir = '/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_matrix_triplets'
        os.makedirs(output_dir, exist_ok=True)
        generate_triplets(output_dir=output_dir,
                          is_log=False, is_min_max=False)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
