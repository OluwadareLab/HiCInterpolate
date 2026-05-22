import os
import re

ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets"
OUTPUT_ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset"

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128, 256, 512]
CHROMOSOMES = {
    "human": ["11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22", "X", "Y"],
    "mouse": ["11", "12", "13", "14", "15", "16",
              "17", "18", "19", "X", "Y"]
}

DATASET = {
    "human": {
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
            "2": {
                "triplets":
                [
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
        input_file = f"{OUTPUT_ROOT_PATH}/dataset_dict_{resolution}.txt"
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
