import os
import re

ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets"
OUTPUT_ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/mm_triplets_dataset"

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128]
CHROMOSOMES = {
    "human": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22", "X", "Y"]
}

# DATASET = {
#     "human": {
#         "dmso": {
#             "control": {
#                 "triplets":
#                 [
#                     ["4DNFIP9EJSOM_dmso_control_0m",
#                      "4DNFI7T93SHL_dmso_control_30m",
#                      "4DNFICF2Z2TG_dmso_control_60m"],

#                     ["4DNFI7T93SHL_dmso_control_30m",
#                      "4DNFICF2Z2TG_dmso_control_60m",
#                      "4DNFILL624WG_dmso_control_90m"],

#                     ["4DNFICF2Z2TG_dmso_control_60m",
#                      "4DNFILL624WG_dmso_control_90m",
#                      "4DNFIC4GB8UM_dmso_control_120m"]
#                 ]
#             }
#         },

#         "hct116": {
#             "1": {
#                 "triplets":
#                 [
#                     ["4DNFIDBFENL7_hct116_1_20m",
#                      "4DNFI9ZUXG61_hct116_1_40m",
#                      "4DNFIAUMRM2S_hct116_1_60m"],

#                     ["4DNFIV56OFE3_hct116_1_auxin_20m",
#                      "4DNFIBCIA62Q_hct116_1_auxin_40m",
#                      "4DNFIQRTP7NM_hct116_1_auxin_60m"]
#                 ]
#             }
#         },

#         "hela_s3": {
#             "r1": {
#                 "triplets":
#                 [
#                     ["4DNFIZZ77KD2_hela_s3_r1_30m",
#                      "4DNFIOLO226X_hela_s3_r1_60m",
#                      "4DNFIJMS2ODT_hela_s3_r1_90m"],

#                     ["4DNFIJMS2ODT_hela_s3_r1_90m",
#                      "4DNFI49F3LJ4_hela_s3_r1_105m",
#                      "4DNFI65MQOIJ_hela_s3_r1_120m"],

#                     ["4DNFI49F3LJ4_hela_s3_r1_105m",
#                      "4DNFI65MQOIJ_hela_s3_r1_120m",
#                      "4DNFIM4KEPRD_hela_s3_r1_135m"],

#                     ["4DNFI65MQOIJ_hela_s3_r1_120m",
#                      "4DNFIM4KEPRD_hela_s3_r1_135m",
#                      "4DNFIIXBIZFC_hela_s3_r1_150m"],

#                     ["4DNFIM4KEPRD_hela_s3_r1_135m",
#                      "4DNFIIXBIZFC_hela_s3_r1_150m",
#                      "4DNFIWDOOBVE_hela_s3_r1_165m"],

#                     ["4DNFIIXBIZFC_hela_s3_r1_150m",
#                      "4DNFIWDOOBVE_hela_s3_r1_165m",
#                      "4DNFIDT9EB5M_hela_s3_r1_180m"],

#                     ["4DNFIWDOOBVE_hela_s3_r1_165m",
#                      "4DNFIDT9EB5M_hela_s3_r1_180m",
#                      "4DNFIX2VUNV8_hela_s3_r1_195m"],

#                     ["4DNFIDT9EB5M_hela_s3_r1_180m",
#                      "4DNFIX2VUNV8_hela_s3_r1_195m",
#                      "4DNFIEQHTV1R_hela_s3_r1_210m"],

#                     ["4DNFIEQHTV1R_hela_s3_r1_210m",
#                      "4DNFIFW7GA64_hela_s3_r1_240m",
#                      "4DNFIXGXD67I_hela_s3_r1_270m"],

#                     ["4DNFIFW7GA64_hela_s3_r1_240m",
#                      "4DNFIXGXD67I_hela_s3_r1_270m",
#                      "4DNFIA7GB1NB_hela_s3_r1_300m"]
#                 ]
#             },

#             "r2": {
#                 "triplets":
#                 [
#                     ["4DNFIX6ZXCA8_hela_s3_r2_30m",
#                      "4DNFIEVR81FS_hela_s3_r2_60m",
#                      "4DNFIAUI6BBI_hela_s3_r2_90m"],

#                     ["4DNFIEVR81FS_hela_s3_r2_60m",
#                      "4DNFIAUI6BBI_hela_s3_r2_90m",
#                      "4DNFIAFEE9G2_hela_s3_r2_120m"],

#                     ["4DNFIAUI6BBI_hela_s3_r2_90m",
#                      "4DNFIAFEE9G2_hela_s3_r2_120m",
#                      "4DNFIPZBEXCP_hela_s3_r2_150m"],

#                     ["4DNFIAFEE9G2_hela_s3_r2_120m",
#                      "4DNFIPZBEXCP_hela_s3_r2_150m",
#                      "4DNFIWPKRZGU_hela_s3_r2_180m"],

#                     ["4DNFIPZBEXCP_hela_s3_r2_150m",
#                      "4DNFIWPKRZGU_hela_s3_r2_180m",
#                      "4DNFIMD9QNDX_hela_s3_r2_210m"],

#                     ["4DNFIWPKRZGU_hela_s3_r2_180m",
#                      "4DNFIMD9QNDX_hela_s3_r2_210m",
#                      "4DNFIATA1HD5_hela_s3_r2_240m"],

#                     ["4DNFIMD9QNDX_hela_s3_r2_210m",
#                      "4DNFIATA1HD5_hela_s3_r2_240m",
#                      "4DNFIH9U4I7I_hela_s3_r2_270m"],

#                     ["4DNFIATA1HD5_hela_s3_r2_240m",
#                      "4DNFIH9U4I7I_hela_s3_r2_270m",
#                      "4DNFIZ95S6TR_hela_s3_r2_300m"]
#                 ]
#             },

#             "r3": {
#                 "triplets":
#                 [
#                     ["4DNFICFZGFAV_hela_s3_r3_30m",
#                      "4DNFIQXCZVVA_hela_s3_r3_60m",
#                      "4DNFIB6PJFJ3_hela_s3_r3_90m"],

#                     ["4DNFIB6PJFJ3_hela_s3_r3_90m",
#                      "4DNFIX97731O_hela_s3_r3_105m",
#                      "4DNFIYQYZOTO_hela_s3_r3_120m"],

#                     ["4DNFIX97731O_hela_s3_r3_105m",
#                      "4DNFIYQYZOTO_hela_s3_r3_120m",
#                      "4DNFIPXU7V25_hela_s3_r3_135m"],

#                     ["4DNFIYQYZOTO_hela_s3_r3_120m",
#                      "4DNFIPXU7V25_hela_s3_r3_135m",
#                      "4DNFIL39PR76_hela_s3_r3_150m"],

#                     ["4DNFIPXU7V25_hela_s3_r3_135m",
#                      "4DNFIL39PR76_hela_s3_r3_150m",
#                      "4DNFIYLJ3R3B_hela_s3_r3_165m"],

#                     ["4DNFIL39PR76_hela_s3_r3_150m",
#                      "4DNFIYLJ3R3B_hela_s3_r3_165m",
#                      "4DNFIL51WBN6_hela_s3_r3_180m"],

#                     ["4DNFIYLJ3R3B_hela_s3_r3_165m",
#                      "4DNFIL51WBN6_hela_s3_r3_180m",
#                      "4DNFI6SFPUDA_hela_s3_r3_195m"],

#                     ["4DNFIL51WBN6_hela_s3_r3_180m",
#                      "4DNFI6SFPUDA_hela_s3_r3_195m",
#                      "4DNFI2KM22QR_hela_s3_r3_210m"],

#                     ["4DNFI2KM22QR_hela_s3_r3_210m",
#                      "4DNFIVF8Q45U_hela_s3_r3_240m",
#                      "4DNFI2RN3WFP_hela_s3_r3_270m"],

#                     ["4DNFIVF8Q45U_hela_s3_r3_240m",
#                      "4DNFI2RN3WFP_hela_s3_r3_270m",
#                      "4DNFI4TJTL7A_hela_s3_r3_300m"]
#                 ]
#             }
#         }
#     }
# }

DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                [
                    ["4DNFIP9EJSOM_dmso_control_0m",
                     "4DNFI7T93SHL_dmso_control_30m",
                     "4DNFICF2Z2TG_dmso_control_60m"],

                    ["4DNFI7T93SHL_dmso_control_30m",
                     "4DNFICF2Z2TG_dmso_control_60m",
                     "4DNFILL624WG_dmso_control_90m"],

                    ["4DNFICF2Z2TG_dmso_control_60m",
                     "4DNFILL624WG_dmso_control_90m",
                     "4DNFIC4GB8UM_dmso_control_120m"]
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
            output_file = f"{OUTPUT_ROOT_PATH}/train_{resolution}_{patch}.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            for organism, samples in DATASET.items():
                for sample, subsamples in samples.items():
                    for subsample, content in subsamples.items():
                        for triplet in content["triplets"]:
                            uuid = triplet[0] + "_" + \
                                triplet[1] + "_" + triplet[2]
                            for chromosome in CHROMOSOMES[organism]:
                                record = f"{str(resolution)}/{str(patch)}/{organism}/{sample}/{subsample}/{str(uuid)}/{chromosome}"
                                get_triplet_dict(
                                    input_file, output_file, record)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
