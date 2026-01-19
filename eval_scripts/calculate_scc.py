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


BASE_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/hicgnn_structures"
DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# REGIONS = [[8000, 8500], [8000, 8500]]
# REGIONS = [
#     [5000, 5500],
#     [5000, 5500]
# ]
# REGIONS = [[2000, 2500], [2000, 2500]]
# REGIONS = [
#     [3000, 3500],
#     [3000, 3500]
# ]

REGIONS = [
    [2500, 3000],
    [2500, 3000]
]

if __name__ == "__main__":
    fieldnames = ['dataset', 'chromosome', 'region', 'scc']
    csv_file = f"{BASE_PATH}/hicgnn_scc_scores.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames)
            writer.writeheader()

    for (dataset, rel_path), region in zip(DATASETS.items(), REGIONS):
        for chrom in CHROMOSOMES:
            
            y_dir = f"{BASE_PATH}/{rel_path}/chr{chrom}_y"
            yt_dir = f"{BASE_PATH}/{rel_path}/chr{chrom}_yt"

            y = f'{y_dir}/chr{chrom}_y_{region[0]}_{region[1]}_structure.pdb'
            yt = f'{yt_dir}/chr{chrom}_yt_{region[0]}_{region[1]}_structure.pdb'

            
            try:
                print(
                f"Processing: {rel_path}_chr{chrom}_{region[0]}_{region[1]}")
                correlation = calculate_spearman_correlation(y, yt)
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
                print(f"Error processing: {rel_path}_chr{chrom}_{region[0]}_{region[1]}")
                traceback.print_exc()

