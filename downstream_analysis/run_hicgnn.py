import os
import subprocess
import sys
import traceback
import numpy as np

MATRIX_BASE_INPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"

DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
RESOLUTIONS = [10000]
# REGIONS = [[6000, 7000], [3000, 4000]]
# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["yt", "y"]
REGIONS = [[6000, 7000]]
CHROMOSOMES = [11]
FILE_SUFFIX = ["y0", "y1"]

# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["y0", "y", "yt", "y1"]
EPOCHS = 100

OUTPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/hicgnn_structures"
HICGNN_FILENAME = "./hicgnn/hicgnn.py"

if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.join("./hicgnn/Data"), exist_ok=True)
    os.makedirs(os.path.join("./hicgnn/Outputs"), exist_ok=True)
    subprocess.run("cd ./hicgnn/Data && rm -rf *", shell=True, check=True)
    subprocess.run("cd ./hicgnn/Outputs && rm -rf *", shell=True, check=True)

    for dataset, rel_path in DATASETS.items():
        for chrom, region in zip(CHROMOSOMES, REGIONS):
            for res in RESOLUTIONS:
                in_sub_dir = os.path.join(MATRIX_BASE_INPUT_PATH, rel_path)
                out_sub_dir = os.path.join(OUTPUT_PATH, rel_path)
                os.makedirs(out_sub_dir, exist_ok=True)
                for suffix in FILE_SUFFIX:
                    out_dir = os.path.join(
                        out_sub_dir, f"chr{chrom}_{suffix}")
                    os.makedirs(out_dir, exist_ok=True)

                    try:
                        full_matrix_file = os.path.join(
                            in_sub_dir, f"chr{chrom}_{suffix}.txt")
                        full_matrix = np.loadtxt(full_matrix_file)

                        matrix = full_matrix[region[0]
                            :region[1], region[0]:region[1]]
                        matrix_file = os.path.join(
                            out_dir, f"chr{chrom}_{suffix}_{region[0]}_{region[1]}.txt")
                        np.savetxt(matrix_file, matrix, fmt="%.6f")

                        cmd = [
                            sys.executable,
                            HICGNN_FILENAME,
                            f"{matrix_file}",
                            "-o", f"{out_dir}",
                            "-ep", str(EPOCHS)
                        ]
                        print(f"Processing {matrix_file}")
                        subprocess.run(cmd, check=True)
                        print(f"Completed {matrix_file}")
                    except Exception as ex:
                        print(f"Exception\n: {ex}")
                        traceback.print_exc()
                    finally:
                        print(
                            f"Removing unnecessary data from Data and Outputs folder")
                        subprocess.run("cd ./hicgnn/Data && rm -rf *",
                                       shell=True, check=True)
                        subprocess.run("cd ./hicgnn/Outputs && rm -rf *",
                                       shell=True, check=True)
