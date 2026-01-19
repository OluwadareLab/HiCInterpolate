import os
import numpy as np
import subprocess
import sys

# BASE_INPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/raw_predicted_data"
# DATASETS = {
#     "dmso_control": "dmso_control",
#     # "dtag_v1": "dtag_v1",
#     # "hct116_noatp30": "hct116_noatp30m",
#     # "hct116_notranscription60m": "hct116_notranscription60m",
#     "hela_s3": "hela_s3",
# }

# CHROMOSOMES = [11, 21]
# BIN_SIZE = 10_000

# BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/embedtad/results"
# os.makedirs(BASE_OUTPUT, exist_ok=True)


def run_embedtad(matrix_file, output_dir, bin_size):
    cmd = [
        sys.executable,
        "/home/hc0783.unt.ad.unt.edu/workspace/codebase/EmbedTAD/embedtad.py",
        "--input", f"{matrix_file}",
        "--output", f"{output_dir}",
        "--resolution", f"{bin_size}",
        "--worker", "GPU",
        "--normalization", "True"
    ]
    subprocess.run(cmd, check=True)


# for dataset, rel_path in DATASETS.items():
#     for chrom in CHROMOSOMES:
#         name = f"{rel_path}_chr{chrom}"
#         print(f"\n🚀 Processing {name}")

#         y_npy = os.path.join(BASE_INPUT, name, f"y_chr{chrom}.npy")
#         hici_npy = os.path.join(BASE_INPUT, name, f"hici_chr{chrom}.npy")

#         y = np.load(y_npy)
#         _max = np.max(y)

#         out_dir = os.path.join(BASE_OUTPUT, name, "y")
#         os.makedirs(out_dir, exist_ok=True)
#         run_embedtad(y_npy, out_dir, "y", BIN_SIZE)

#         out_dir = os.path.join(BASE_OUTPUT, name, "hici")
#         os.makedirs(out_dir, exist_ok=True)
#         run_embedtad(hici_npy, out_dir, "hici", BIN_SIZE, _max)

# print("\n✅ All EmbedTAD runs completed.")


MATRIX_BASE_INPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"

DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
RESOLUTIONS = [10000]
# REGIONS = [[4750, 5250]]
CHROMOSOMES = [11, 13, 15, 17, 19, 21]
FILE_SUFFIX = ["yt", "y"]
# REGIONS = [[6000, 7000]]
# CHROMOSOMES = [11]
# FILE_SUFFIX = ["y0", "yt", "y", "y1"]

# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["y0", "y", "yt", "y1"]
EPOCHS = 50

OUTPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/embedtad"

if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_PATH), exist_ok=True)

    for dataset, rel_path in DATASETS.items():
        for chrom in CHROMOSOMES:
            for res in RESOLUTIONS:
                in_sub_dir = os.path.join(MATRIX_BASE_INPUT_PATH, rel_path)
                out_sub_dir = os.path.join(OUTPUT_PATH, rel_path)
                os.makedirs(out_sub_dir, exist_ok=True)
                for suffix in FILE_SUFFIX:
                    out_dir = os.path.join(
                        out_sub_dir, f"chr{chrom}_{suffix}")
                    os.makedirs(out_dir, exist_ok=True)
                    matrix = os.path.join(
                        in_sub_dir, f"chr{chrom}_{suffix}.txt")
                    run_embedtad(matrix, out_dir, res)
