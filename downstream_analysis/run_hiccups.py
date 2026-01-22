import os
import numpy as np
import subprocess

JAVA_JAR = "./downstream_analysis/juicer_tools.2.20.00.jar"

def matrix_to_contact_list(matrix_file, contact_file, chrom, bin_size):
    _, ext = os.path.splitext(matrix_file)
    print("Extension:", ext)
    if ext.lower() == ".npy":
        mat = np.load(matrix_file)
    if ext.lower() == ".txt":
        mat = np.loadtxt(matrix_file)

    rows, cols = np.triu_indices_from(mat)
    values = mat[rows, cols]

    with open(contact_file, "w") as f:
        for i, j, v in zip(rows, cols, values):
            if v != 0:
                pos1 = i * bin_size
                pos2 = j * bin_size
                f.write(f"chr{chrom} {pos1} chr{chrom} {pos2} {v:.6f}\n")


def run_hiccups(matrix_file, output_dir, bin_size, chrom, genome_id):
    contact_file = os.path.join(output_dir, f"chr{chrom}.contacts")
    matrix_to_contact_list(matrix_file=matrix_file, contact_file=contact_file,
                           chrom=chrom, bin_size=bin_size)

    hic_file = os.path.join(output_dir, f"chr{chrom}.hic")
    cmd_pre = [
        "java", "-jar", JAVA_JAR,
        "pre",
        contact_file,
        hic_file,
        genome_id,
        "-r", str(bin_size),
        "-c", f"chr{chrom}"
    ]
    print("Running juicer_tools pre...")
    subprocess.run(cmd_pre, check=True)
    print(f".hic generated: {hic_file}")
    loops_output_dir = os.path.join(output_dir, "loops")
    os.makedirs(loops_output_dir, exist_ok=True)
    cmd_hiccups = [
        "java", "-jar", JAVA_JAR,
        "hiccups",
        "--cpu",
        "--threads", "40",
        "-c", f"chr{chrom}",
        "-r", str(bin_size),
        hic_file,
        loops_output_dir
    ]
    print("Running juicer_tools hiccups...")
    subprocess.run(cmd_hiccups, check=True)
    print(f"HICCUPS loops saved in {output_dir}")
