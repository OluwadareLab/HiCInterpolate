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


def txt_to_np3_matrix(txt_matrix_filename: str, res: int, chr: int, output_filename: str):
    contact_matrix = pd.read_csv(
        txt_matrix_filename, delimiter='\t', header=None)
    N = contact_matrix.shape[0]
    start_positions = np.arange(0, res * N, res)
    end_positions = start_positions + res
    chrom = [chr] * N
    topdom_data = pd.DataFrame({
        "chrom": chrom,
        "start": start_positions,
        "end": end_positions
    })
    np3_df = pd.concat([topdom_data, contact_matrix], axis=1)
    np3_df.to_csv(output_filename, sep='\t', index=False, header=False)
    print(f"n x (n+3) data written to {output_filename}")


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
                    np3_filename = f"{out_sub_dir}/{chrom}/chr{chrom}_{suffix}.np3"
                    txt_to_np3_matrix(
                        txt_matrix_filename=file_path, res=res, chr=chrom, output_filename=np3_filename)
