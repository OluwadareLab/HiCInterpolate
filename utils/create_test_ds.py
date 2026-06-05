import os
import re

ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets"
DATASET_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/log_mm_triplets_dataset"
OUTPUT_ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/log_mm_triplets_dataset/test"

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128]
CHROMOSOMES = {
    "human": ["11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22", "X", "Y"],
    "mouse": ["11", "12", "13", "14", "15", "16",
              "17", "18", "19", "X", "Y"]
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
        "hct116": {
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
                    ["4DNFIPZBEXCP_hela_s3_r2_150m",
                     "4DNFIWPKRZGU_hela_s3_r2_180m",
                     "4DNFIMD9QNDX_hela_s3_r2_210m"]
                ]
            }
        }
    },
    "mouse": {
        "embryo": {
            "development": {
                "triplets": [
                    ["4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell"],

                    ["4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm",
                     "4DNFI5IAH9H1_mes_cell"]
                ]
            }
        }
    }
}


def get_triplet_dict(input_file, output_file, regex_pattern):
    pattern = re.compile(regex_pattern)
    with open(input_file, "r") as infile, open(output_file, "a") as outfile:
        for line in infile:
            line = line.strip()
            if pattern.search(line):
                outfile.write(line + "\n")

        infile.flush()
        infile.close()
        outfile.flush()
        outfile.close()


def prepare_triplates():
    for resolution in RESOLUTIONS:
        input_file = f"{DATASET_PATH}/dataset_dict_{resolution}.txt"
        for patch in PATCHES:
            for organism, samples in DATASET.items():
                for sample, subsamples in samples.items():
                    for subsample, content in subsamples.items():
                        for triplet in content["triplets"]:
                            uuid = triplet[0] + "_" + \
                                triplet[1] + "_" + triplet[2]
                            for chromosome in CHROMOSOMES[organism]:
                                record = f"{str(resolution)}/{str(patch)}/{organism}/{sample}/{subsample}/{str(uuid)}/{chromosome}"

                                output_file = f"{OUTPUT_ROOT_PATH}/test_{resolution}_{patch}_{organism}_{uuid}_{chromosome}.txt"
                                os.makedirs(os.path.dirname(
                                    output_file), exist_ok=True)

                                get_triplet_dict(
                                    input_file, output_file, record)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
