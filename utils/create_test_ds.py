import os
import re

ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets"
DATASET_DICT_PATH = f"{ROOT_PATH}/triplets_dataset"
OUTPUT_PATH = f"{ROOT_PATH}/triplets_dataset/test"

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128]
CHROMOSOMES = {
    "human": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22"],
    "mouse": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11", "12", "13", "14", "15", "16",
              "17", "18", "19"]
}

DATASET = {
    "human": {
        # unseen
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

        # seen
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
            # unseen replicate
            "1": {
                "triplets":
                [
                    ["4DNFIDBFENL7_hct116_1_20m",
                     "4DNFI9ZUXG61_hct116_1_40m",
                     "4DNFIAUMRM2S_hct116_1_60m"]
                ]
            },

            # seen
            "2": {
                "triplets":
                [
                    ["4DNFIAAH19VM_hct116_2_20m",
                     "4DNFI7QUSU5J_hct116_2_40m",
                     "4DNFIXEB4UZO_hct116_2_60m"]
                ]
            },
        }
    },
    # unseen organism
    "mouse": {
        "embryo": {
            "development": {
                "triplets": [
                    ["4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm"]
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
        input_file = f"{DATASET_DICT_PATH}/dataset_dict_{resolution}.txt"
        for patch in PATCHES:
            for organism, samples in DATASET.items():
                for sample, subsamples in samples.items():
                    for subsample, content in subsamples.items():
                        for triplet in content["triplets"]:
                            uuid = triplet[0] + "_" + \
                                triplet[1] + "_" + triplet[2]
                            for chromosome in CHROMOSOMES[organism]:
                                record = f"{str(resolution)}/{str(patch)}/{organism}/{sample}/{subsample}/{str(uuid)}/{chromosome}"

                                output_file = f"{OUTPUT_PATH}/test_{resolution}_{patch}_{organism}_{triplet[1]}_{chromosome}.txt"
                                os.makedirs(os.path.dirname(
                                    output_file), exist_ok=True)

                                get_triplet_dict(
                                    input_file, output_file, record)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
