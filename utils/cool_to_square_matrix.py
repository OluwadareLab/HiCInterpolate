import os
import numpy as np
import cooler
import matplotlib.pyplot as plt
from random import seed
from dataset import PATCH_SIZES, STEP_SIZES, RESOLUTIONS, DATASET, INPUT_DIR, CMAP

seed(42)


def normalize(mat, is_log=False, is_min_max=False):
    if is_log:
        mat = np.log1p(mat)

    if is_min_max:
        min_val = np.min(mat)
        max_val = np.max(mat)
        denom = max_val - min_val
        if denom == 0:
            fill_value = 1.0 if min_val > 0 else 0.0
            mat = np.full_like(mat, fill_value, dtype=np.float32)
        else:
            mat = (mat - min_val) / denom

    return mat


def plot_patch(img1, img2, img3, filename):
    data_groups = [img1, img2, img3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = np.atleast_2d(axes)

    for idx in range(len(data_groups)):
        ax = axes[0, idx]
        matrix = data_groups[idx]
        _min = np.min(matrix)
        _max = np.max(matrix)
        im = ax.imshow(matrix, cmap=CMAP, vmin=_min, vmax=_max)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, format='png')
    plt.close()


def save_patch(matrix1, matrix2, matrix3, chrom_size, output_dir, out_sub_dir, ds_dict_filename):
    with open(ds_dict_filename, "a") as dict_file:
        for patch_size, step_size in zip(PATCH_SIZES, STEP_SIZES):
            print(
                f"[INFO] Saving patches for {out_sub_dir} with patch size {patch_size} and step size {step_size}")

            ori_bins = matrix2.shape[0]
            for start_bin in range(0, ori_bins - patch_size + 1, step_size):
                end_bin = start_bin + patch_size

                patch1 = matrix1[start_bin:end_bin, start_bin:end_bin]
                patch2 = matrix2[start_bin:end_bin, start_bin:end_bin]
                patch3 = matrix3[start_bin:end_bin, start_bin:end_bin]

                if patch1.shape == (patch_size, patch_size) and patch2.shape == (patch_size, patch_size) and patch3.shape == (patch_size, patch_size):
                    sub_dir = f"{out_sub_dir}/{patch_size}/{start_bin}_{end_bin}"
                    dict_file.write(f"{sub_dir}\n")
                    os.makedirs(os.path.join(
                        output_dir, sub_dir), exist_ok=True)

                    np.save(os.path.join(
                        output_dir, sub_dir, 'img1.npy'), patch1)
                    np.save(os.path.join(
                        output_dir, sub_dir, 'img2.npy'), patch2)
                    np.save(os.path.join(
                        output_dir, sub_dir, 'img3.npy'), patch3)

            plot_out_dir = os.path.join(
                output_dir, 'plots', out_sub_dir, str(patch_size))
            os.makedirs(plot_out_dir, exist_ok=True)

            plot_filename = os.path.join(
                plot_out_dir, f'{start_bin}_{end_bin}.png')
            plot_patch(patch1, patch2, patch3, plot_filename)


def generate_triplets(output_dir, is_log=False, is_min_max=False):
    for resolution in RESOLUTIONS:
        for organism, samples in DATASET.items():
            for sample, conditions in samples.items():
                for condition, dataset in conditions.items():
                    for triplet in dataset["triplets"]:
                        uuid = triplet[0].split(
                            "_")[-1] + "_" + triplet[1].split("_")[-1] + "_" + triplet[2].split("_")[-1]

                        filename1 = f"{INPUT_DIR}/{triplet[0]}_{resolution}_KR.cool"
                        filename2 = f"{INPUT_DIR}/{triplet[1]}_{resolution}_KR.cool"
                        filename3 = f"{INPUT_DIR}/{triplet[2]}_{resolution}_KR.cool"

                        if not (os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3)):
                            print(
                                f"[WARNING] Missing files for {organism} > {sample} > {condition} > {uuid} > {resolution}")
                            continue

                        cool1 = cooler.Cooler(
                            f"{INPUT_DIR}/{triplet[0]}_{resolution}_KR.cool")
                        cool2 = cooler.Cooler(
                            f"{INPUT_DIR}/{triplet[1]}_{resolution}_KR.cool")
                        cool3 = cooler.Cooler(
                            f"{INPUT_DIR}/{triplet[2]}_{resolution}_KR.cool")

                        ds_dict_filename = f"{output_dir}/dataset_dict_{resolution}.txt"
                        os.makedirs(os.path.dirname(
                            ds_dict_filename), exist_ok=True)

                        for chrom in cool2.chromnames:
                            print(
                                f"[INFO] Processing {organism} > {sample} > {condition} > {uuid} > {resolution} > chr{chrom}")

                            out_sub_dir = f'{resolution}/{organism}/{sample}/{condition}/{triplet[1]}/chr{chrom}'

                            matrix1 = normalize(cool1.matrix(balance=False).fetch(
                                chrom), is_log=is_log, is_min_max=is_min_max)
                            matrix2 = normalize(cool2.matrix(balance=False).fetch(
                                chrom), is_log=is_log, is_min_max=is_min_max)
                            matrix3 = normalize(cool3.matrix(balance=False).fetch(
                                chrom), is_log=is_log, is_min_max=is_min_max)

                            save_patch(
                                matrix1, matrix2, matrix3, cool2.chromsizes[chrom], output_dir, out_sub_dir, ds_dict_filename)


if __name__ == "__main__":
    try:
        # log -> min-max
        # output_dir = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/log_min_max_triplets'
        # os.makedirs(output_dir, exist_ok=True)
        # generate_triplets(output_dir=output_dir, is_log=True, is_min_max=True)

        # # min-max
        # output_dir = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/min_max_triplets'
        # os.makedirs(output_dir, exist_ok=True)
        # generate_triplets(output_dir=output_dir, is_log=False, is_min_max=True)

        # none
        output_dir = '/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/new_triplets'
        os.makedirs(output_dir, exist_ok=True)
        generate_triplets(output_dir=output_dir,
                          is_log=False, is_min_max=False)

        # # log
        # output_dir = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/log_triplets'
        # os.makedirs(output_dir, exist_ok=True)
        # generate_triplets(output_dir=output_dir, is_log=True, is_min_max=False)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
