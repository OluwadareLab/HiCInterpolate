import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
import math
import os
import csv
import traceback


def extract_coordinates(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.coord)
    print("Sample XYZ coordinates:", coords[1])

    return np.array(coords)


def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def structure_distance(coordinate):
    n = len(coordinate)
    distance_matrix = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = euclidean_distance(coordinate[i], coordinate[j])
            # print(dist)
            distance_matrix[i, j] = dist
            # Since the distance matrix is symmetric
            distance_matrix[j, i] = dist
    return distance_matrix


def calculate_spearman_correlation(pdb_file1, pdb_file2):
    coords1 = extract_coordinates(pdb_file1)
    coords2 = extract_coordinates(pdb_file2)
    length1 = len(coords1)
    length2 = len(coords2)

    print(length1)
    print("length coodrinate 1:")
    # print(coords1)
    print(length2)

    dist1 = structure_distance(coords1)
    length3 = len(dist1)
    print(length3)

    # print(dist1)
    dist2 = structure_distance(coords2)
    length4 = len(dist2)
    print(length4)

    # Flatten the distance matrices
    dist1_flat = dist1.flatten()
    dist2_flat = dist2.flatten()

    correlation, _ = spearmanr(dist1_flat, dist2_flat)
    return correlation


BASE_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiC-GNN/output_main"
DATASETS = {
    "dmso_control": "dmso_control",
    "dtag_v1": "dtag_v1",
    "hct116_noatp30": "hct116_noatp30m",
    "hct116_notranscription60m": "hct116_notranscription60m",
    "hela_s3": "hela_s3"
}
GENOME_ID = "hg38"
BIN_SIZE = 10_000

CHROMOSOMES = [11, 21]
CHROMOSOME_SIZE = [135086622, 46709983]
# REGIONS = [[3000, 3050], [1500, 1550]]
REGIONS = [[4100, 4200], [1500, 1600]]


if __name__ == "__main__":
    fieldnames = ['dataset', 'chromosome', 'region', 'scc']
    csv_file = f"{BASE_PATH}/hicgnn_scc_scores.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames)
            writer.writeheader()

    for dataset, rel_path in DATASETS.items():
        for chrom, chrom_size, region in zip(CHROMOSOMES, CHROMOSOME_SIZE, REGIONS):
            try:
                dir = f"{BASE_PATH}/{rel_path}_chr{chrom}_{region[0]}_{region[1]}"

                y = f'{dir}/y_chr{chrom}_{region[0]}_{region[1]}_structure.pdb'
                hici = f'{dir}/hici_chr{chrom}_{region[0]}_{region[1]}_structure.pdb'

                print(
                    f"Processing: {rel_path}_chr{chrom}_{region[0]}_{region[1]}")
                correlation = calculate_spearman_correlation(y, hici)
                print(f"Spearman correlation coefficient: {correlation}")
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(
                        f, fieldnames=fieldnames)
                    writer.writerow({
                        'dataset': dataset,
                        'chromosome': chrom,
                        'region': f'{region[0]}-{region[1]}',
                        'scc': correlation
                    })
                    f.close()
            except Exception as ex:
                print(f"Exception\n: {ex}")
                traceback.print_exc()
