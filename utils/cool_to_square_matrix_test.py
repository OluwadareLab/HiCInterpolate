import os
import numpy as np
import cooler as cool

ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets"
OUTPUT_ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset/test"

RESOLUTIONS = [25000, 10000, 5000]
BALANCE_COOL = False
PATCHES = [64, 128, 256, 512]
CLIPPING_PERCENTILE = 99.99
PATCH_OVERLAP_RATIO = 0.2

COUTER = {
    5000: {64: 0, 128: 0, 256: 0, 512: 0},
    10000: {64: 0, 128: 0, 256: 0, 512: 0},
    25000: {64: 0, 128: 0, 256: 0, 512: 0}
}

CHROMOSOMES = ["11", "12", "13", "14", "15", "16",
               "17", "18", "19", "20", "21", "22", "X", "Y"]

TEST_DATASET = {
    "human": {
        # Unseen samples for testing
        "dtag": {
            "v1": {
                "0m_30m_60m":
                    ["4DNFI5EAPQTI_dtag_v1_0m",
                     "4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m"],
                "30m_60m_90m":
                    ["4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m"],

                "60m_90m_120m":
                    ["4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m",
                     "4DNFIPZCCTV6_dtag_v1_120m"]
            }
        },
        "hct116": {
            # Trained on set 1, test on set 2 unseen samples
            "2": {
                "20m_40m_60m":
                    ["4DNFIAAH19VM_hct116_2_20m",
                     "4DNFI7QUSU5J_hct116_2_40m",
                     "4DNFIXEB4UZO_hct116_2_60m"],
                "no_transcription_360m_20m_40m_60m":
                    ["4DNFI5IZNXIO_hct116_2_no_transcription_360m_20m",
                     "4DNFIZK7W8GZ_hct116_2_no_transcription_360m_40m",
                     "4DNFISRP84FE_hct116_2_no_transcription_360m_60m"],
                "no_transcription_60m_20m_40m_60m":
                    ["4DNFII16KXA7_hct116_2_no_transcription_60m_20m",
                     "4DNFIMIMLMD3_hct116_2_no_transcription_60m_40m",
                     "4DNFI2LY7B73_hct116_2_no_transcription_60m_60m"],
                "no_atp_120m_20m_40m_60m":
                    ["4DNFITUPI4HA_hct116_2_no_atp_120m_20m",
                     "4DNFIM7Q2FQQ_hct116_2_no_atp_120m_40m",
                     "4DNFISATK9PF_hct116_2_no_atp_120m_60m"],
                "no_atp_30m_20m_40m_60m":
                    ["4DNFIVC8OQPG_hct116_2_no_atp_30m_20m",
                     "4DNFI44JLUSL_hct116_2_no_atp_30m_40m",
                     "4DNFIBED48O1_hct116_2_no_atp_30m_60m"],
                "no_replication_20m_40m_60m":
                    ["4DNFIDD9IF9T_hct116_2_no_replication_20m",
                     "4DNFIQWWATGK_hct116_2_no_replication_40m",
                     "4DNFI3NTD7B3_hct116_2_no_replication_60m"]
            }
        }
    },
    # Unseen organizm and samples for testing
    "mouse": {
        "embryo": {
            "development": {
                "sperm_mii_oocyte_zygote":
                    ["4DNFIN8F14CS_sperm",
                     "4DNFIVCJKHMN_mii_oocyte",
                     "4DNFI1EYIGOC_zygote"],
                "mii_oocyte_zygote_early2_cell":
                    ["4DNFIVCJKHMN_mii_oocyte",
                     "4DNFI1EYIGOC_zygote",
                     "4DNFIK4CECUH_early2_cell"],
                "zygote_early2_cell_late2_cell":
                    ["4DNFI1EYIGOC_zygote",
                     "4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell"],
                "early2_cell_late2_cell_8cell":
                    ["4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell"],
                "late2_cell_8cell_icm":
                    ["4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm"],
                "8cell_icm_mes_cell":
                    ["4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm",
                     "4DNFI5IAH9H1_mes_cell"]
            }
        }
    }
}


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch].astype(np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)


def reset_counter(counter):
    for outer_key in counter:
        for inner_key in counter[outer_key]:
            counter[outer_key][inner_key] = 0


def generate_patch(mat_0, mat_y, mat_1, organism, sample, subsample, resolution, chromosome, output_root_path, uuid):
    for patch in PATCHES:
        patch_path = f"{output_root_path}/{patch}/{chromosome}/{uuid}"
        os.makedirs(patch_path, exist_ok=True)
        ds_dict_file = f"{patch_path}/dataset_dict_{resolution}.txt"
        os.makedirs(os.path.dirname(ds_dict_file), exist_ok=True)

        reset_counter(COUTER)

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
                    COUTER[resolution][patch] += 1


def prepare_triplates():
    for resolution in RESOLUTIONS:
        res_path = f"{OUTPUT_ROOT_PATH}/{resolution}"
        os.makedirs(res_path, exist_ok=True)
        for organism, samples in TEST_DATASET.items():
            org_path = f"{res_path}/{organism}"
            os.makedirs(org_path, exist_ok=True)
            for sample, subsamples in samples.items():
                for subsample, content in subsamples.items():
                    for key, triplet in content.items():
                        print(
                            f"Processing {organism} > {sample} > {subsample} > {resolution} > {triplet}")

                        filepath0 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[0]}_{resolution}_KR.cool"
                        filepath1 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[1]}_{resolution}_KR.cool"
                        filepath2 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[2]}_{resolution}_KR.cool"

                        cool_0 = cool.Cooler(filepath0)
                        cool_y = cool.Cooler(filepath1)
                        cool_1 = cool.Cooler(filepath2)
                        uuid = key
                        for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                            if chromosome not in CHROMOSOMES:
                                continue

                            fetch = f"{chromosome}:{0}-{chr_size}"
                            mat_0 = cool_0.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            mat_y = cool_y.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            mat_1 = cool_1.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            generate_patch(mat_0=mat_0, mat_y=mat_y, mat_1=mat_1,
                                           organism=organism, sample=sample, subsample=subsample, resolution=resolution, chromosome=chromosome, output_root_path=org_path, uuid=uuid)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
