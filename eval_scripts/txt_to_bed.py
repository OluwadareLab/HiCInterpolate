import os
import numpy as np
import pandas as pd


BASE_INPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"
DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
RESOLUTIONS = [10000]
CHROMOSOMES = [11, 13, 15, 17, 19, 21]
FILE_SUFFIX = ["y", "yt"]
BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"


def matrix_to_contact_list(matrix_file, contact_file, chrom, bin_size):
    mat = np.loadtxt(matrix_file)
    mat = np.rint(mat).astype(int)
    df = pd.DataFrame(mat)

    start_pos = 0
    bins = np.arange(start_pos, start_pos + bin_size * len(df), bin_size)

    # Assign row and column labels
    df.index = bins
    df.columns = bins
    df.to_csv(contact_file, sep="\t", index=False, header=False)

    # rows, cols = np.triu_indices_from(mat)
    # values = mat[rows, cols]

    # with open(contact_file, "w") as f:
    #     f.write(f"region1\tregion2\tIF\n")
    #     for i, j, v in zip(rows, cols, values):
    #         if v != 0:
    #             pos1 = i * bin_size
    #             pos2 = j * bin_size
    #             f.write(f"{pos1}\t{pos2}\t{round(v)}\n")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_OUTPUT), exist_ok=True)
    for dataset, rel_path in DATASETS.items():
        for chrom in CHROMOSOMES:
            for res in RESOLUTIONS:
                in_sub_dir = os.path.join(BASE_INPUT, rel_path)
                out_sub_dir = os.path.join(BASE_OUTPUT, rel_path)
                os.makedirs(out_sub_dir, exist_ok=True)

                for suffix in FILE_SUFFIX:
                    file_path = f"{in_sub_dir}/chr{chrom}_{suffix}.txt"
                    print(f"Processing: {file_path}...")
                    out_dir = os.path.join(out_sub_dir, f"chr{chrom}_{suffix}")
                    os.makedirs(out_sub_dir, exist_ok=True)
                    np3_filename = f"{out_sub_dir}/chr{chrom}_{suffix}.tsv"
                    matrix_to_contact_list(
                        matrix_file=file_path, contact_file=np3_filename, chrom=chrom, bin_size=res)
