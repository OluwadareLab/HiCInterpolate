import os
import numpy as np
import subprocess
import sys


def run_embedtad(matrix_file, output_dir, bin_size):
    _, ext = os.path.splitext(matrix_file)
    print("Extension:", ext)
    mat_file = matrix_file
    if ext.lower() == ".npy":
        data = np.load(matrix_file, allow_pickle=True)
        mat_file = f"{output_dir}/matrix.txt"
        np.savetxt(mat_file, data)

    cmd = [
        sys.executable,
        "./downstream_analysis/EmbedTAD/embedtad.py",
        "--input", f"{mat_file}",
        "--output", f"{output_dir}",
        "--resolution", f"{bin_size}",
        "--worker", "CPU",
        "--normalization", "True"
    ]
    subprocess.run(cmd, check=True)
