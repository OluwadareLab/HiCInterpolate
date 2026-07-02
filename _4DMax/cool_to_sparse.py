import cooler
import os
from matplotlib.pyplot import cool
import pandas as pd
import subprocess

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
            "notranscription60m"
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
                "4DNFI5IZNXIO_hct116_2_notranscription360m_20m",
                "4DNFIZK7W8GZ_hct116_2_notranscription360m_40m",
                "4DNFISRP84FE_hct116_2_notranscription360m_60m"
            ]
        ]
    ]
]

CHROMOSOMES = [21]
OUTPUT_DIR = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/4DMax/data"


for organism, org_samples, org_subsamples, org_filenames in zip(ORGANISMS, SAMPLES, SUBSAMPLES, FILENAME_LIST):
    for sample, sam_sub_sample, sam_sample_filenames in zip(org_samples, org_subsamples, org_filenames):
        for sub_sample, sample_filenames in zip(sam_sub_sample, sam_sample_filenames):
            for resolution in RESOLUTIONS:
                times = [0, 2]
                times[0] = int(sample_filenames[0].split(
                    '_')[-1].replace('m', ''))
                times[1] = int(sample_filenames[2].split(
                    '_')[-1].replace('m', ''))

                timeframe_name = "_".join(name.split(
                    '_')[-1] for name in sample_filenames[:3])

                for chromosome in CHROMOSOMES:
                    sub_dir = f"{OUTPUT_DIR}/{organism}/{sample}/{sub_sample}/{timeframe_name}"
                    os.makedirs(sub_dir, exist_ok=True)
                    print(
                        f"Processing {sub_dir} chr{chromosome} at resolution {resolution}")
                    json_info = {
                        "name": f"{sample}_{sub_sample}",
                        "step": 21,
                        "res": resolution,
                        "chro": chromosome,
                        "rep": 1,
                        "start_t": times[0],
                        "end_t": times[1],
                        "taos": times,
                        "dataset": [
                            f"{resolution}/{chromosome}/chr{chromosome}_{times[0]}",
                            f"{resolution}/{chromosome}/chr{chromosome}_{times[1]}"
                        ]
                    }

                    with open(f"{sub_dir}/chr{chromosome}_info.json", mode="w", encoding="utf-8") as json_file:
                        import json
                        json.dump(json_info, json_file, indent=4)

                    os.makedirs(
                        f"{sub_dir}/{resolution}/{chromosome}", exist_ok=True)

                    command = [
                        "cooler", "dump",
                        "-r", str(chromosome),
                        f"{ROOT_PATH}/time_series_data/{organism}/sample/{sample}/{sample_filenames[0]}_{resolution}_KR.cool"
                    ]
                    with open(f"{sub_dir}/{resolution}/{chromosome}/chr{chromosome}_{times[0]}.txt", "w") as outfile:
                        subprocess.run(command, stdout=outfile, check=True)

                    command = [
                        "cooler", "dump",
                        "-r", str(chromosome),
                        f"{ROOT_PATH}/time_series_data/{organism}/sample/{sample}/{sample_filenames[2]}_{resolution}_KR.cool"
                    ]
                    with open(f"{sub_dir}/{resolution}/{chromosome}/chr{chromosome}_{times[1]}.txt", "w") as outfile:
                        subprocess.run(command, stdout=outfile, check=True)
