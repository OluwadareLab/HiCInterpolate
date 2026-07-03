import os
import re

ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries"
DATASET_DICT_PATH = f"{ROOT_PATH}/new_triplets"
OUTPUT_PATH = f"{ROOT_PATH}/new_triplets/test"

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128, 256, 512]
CHROMOSOMES = {
    "human": ["10", "11", "15", "16", "20", "21"],
    "mouse": ["10", "15", "19"]
}

DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                    [
                        ["dmso_control_0m",
                         "dmso_control_30m",
                         "dmso_control_60m"],

                        ["dmso_control_30m",
                         "dmso_control_60m",
                         "dmso_control_90m"],

                        ["dmso_control_60m",
                         "dmso_control_90m",
                         "dmso_control_120m"]
                    ]
            }
        },
        "dtag": {
            "v1": {
                "triplets":
                    [
                        ["dtag_v1_0m",
                         "dtag_v1_30m",
                         "dtag_v1_60m"],

                        ["dtag_v1_30m",
                         "dtag_v1_60m",
                         "dtag_v1_90m"],

                        ["dtag_v1_60m",
                         "dtag_v1_90m",
                         "dtag_v1_120m"]
                    ]
            }
        },
        "wtc11": {
            "atrial": {
                "triplets":
                    [
                        ["wtc11_atrial_2880m",
                         "wtc11_atrial_5760m",
                         "wtc11_atrial_8640m"]
                    ]
            },
            "ventricular": {
                "triplets":
                    [
                        ["wtc11_ventricular_2880m",
                         "wtc11_ventricular_5760m",
                         "wtc11_ventricular_8640m"]
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
                    ["sperm",
                     "mii_oocyte",
                     "zygote"],

                    ["mii_oocyte",
                     "zygote",
                     "early2_cell"],

                    ["zygote",
                     "early2_cell",
                     "late2_cell"],

                    ["early2_cell",
                     "late2_cell",
                     "8cell"],

                    ["late2_cell",
                     "8cell",
                     "icm"],

                    ["8cell",
                     "icm",
                     "mes_cell"]
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
                            for chromosome in CHROMOSOMES[organism]:
                                record = f"{resolution}/{organism}/{sample}/{subsample}/{triplet[1]}/chr{chromosome}/{patch}"
                                output_file = f"{OUTPUT_PATH}/{resolution}_{patch}_{organism}_{sample}_{subsample}_{triplet[1]}_chr{chromosome}.test"
                                os.makedirs(os.path.dirname(
                                    output_file), exist_ok=True)
                                get_triplet_dict(
                                    input_file, output_file, record)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
