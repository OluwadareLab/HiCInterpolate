import os
import numpy as np
import math
import cooler as cool
import matplotlib.pyplot as plt

ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data"
OUTPUT_PATH = f"{ROOT_PATH}/pairs/kr_log_clip_norm"
RESOLUTIONS = [10000]
BALANCE = True
PATCH_SIZES = [256]

ORGANISMS = ["human"]
ORGANISM_SAMPLES = [["verticular"]]
FILENAME_SET = [
    [[["v_2d_4DNFI1P2EP7L", "v_4d_4DNFI7C5YXNX", "v_6d_4DNFI8I2WYXS"]]]]
CHROMOSOMES = [19]

_CMAP = "YlOrRd"
_EPSILON = 1e-8
PERCENTILE = 99.99


def plot_hic_map(matrix, filename):
    plt.imshow(matrix, cmap=_CMAP)
    plt.title(f"{filename}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, format='png')
    plt.close()


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch]
    submatrix = submatrix.astype(np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)


def get_padded_matrix(_matrix, patch_size):
    h, w = _matrix.shape
    _extension = math.ceil(h/patch_size)
    pad_h = (_extension*patch_size) - h
    pad_w = (_extension*patch_size) - w
    padded = np.pad(_matrix, ((0, pad_h), (0, pad_w)),
                    mode='constant', constant_values=_EPSILON)
    return padded


def generate_patch(mat_1, mat_2, mat_3, organism, sample, resolution, chromosome, ds_filename, sub_sample, counter):
    for patch, i in zip(PATCH_SIZES, range(0, len(counter))):

        orginal_path = f"{OUTPUT_PATH}/{patch}/{ds_filename}/{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}"
        os.makedirs(orginal_path, exist_ok=True)
        np.save(f"{orginal_path}/img_1.npy", mat_1)
        np.save(f"{orginal_path}/img_2.npy", mat_2)
        np.save(f"{orginal_path}/img_3.npy", mat_3)

        orginal_file = f"{OUTPUT_PATH}/{patch}/{ds_filename}_raw.txt"
        os.makedirs(os.path.dirname(orginal_file), exist_ok=True)
        with open(orginal_file, "w") as file:
            file.write(orginal_path+"\n")
            file.close()

        mat_1 = get_padded_matrix(mat_1, patch)
        mat_3 = get_padded_matrix(mat_3, patch)

        ds_file = f"{OUTPUT_PATH}/{patch}/{ds_filename}.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating {ds_filename} patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {resolution} > chr{chromosome}")
            row, col = mat_1.shape
            bin_inc = patch

            r = 0
            while (r+patch <= row):
                c = 0
                while (c+patch <= col):
                    folder = f"{counter[i]:06d}"
                    path = f"{ds_filename}/{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}/{folder}"
                    file.write(path+"\n")
                    path = f"{OUTPUT_PATH}/{patch}/{path}"
                    os.makedirs(path, exist_ok=True)
                    save_img(mat_1, r, c, patch, path, "img1")
                    save_img(mat_3, r, c, patch, path, "img3")
                    counter[i] += 1
                    c += bin_inc
                r += bin_inc
            file.close()
    return counter


def normalization(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    log_matrix = np.log1p(matrix+1)
    percentile_val = np.percentile(log_matrix, PERCENTILE)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)
    norm_matrix = clip_matrix / percentile_val

    return norm_matrix


def generate_ds(ORGANISMS, ORGANISM_SAMPLES, FILENAME_SET, ds_filename):
    for organism, org_samples, org_filenames in zip(ORGANISMS, ORGANISM_SAMPLES, FILENAME_SET):
        for sample, sample_filenames in zip(org_samples, org_filenames):
            for resolution in RESOLUTIONS:
                for filenames in sample_filenames:

                    cool_1 = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/{sample}/{filenames[0]}_{resolution}_KR.cool")
                    cool_2 = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/{sample}/{filenames[1]}_{resolution}_KR.cool")
                    cool_3 = cool.Cooler(
                        f"{ROOT_PATH}/{organism}/{sample}/{filenames[2]}_{resolution}_KR.cool")

                    sub_sample = "_".join(name.split(
                        '_')[-1] for name in filenames[:3])

                    for chromosome in CHROMOSOMES:

                        chr_size = cool_2.chromsizes[f"{chromosome}"]
                        fetch = f"{chromosome}:{0}-{chr_size}"

                        chr_mat_1 = cool_1.matrix(
                            balance=BALANCE).fetch(fetch)
                        chr_mat_1 = normalization(chr_mat_1)

                        chr_mat_2 = cool_2.matrix(
                            balance=BALANCE).fetch(fetch)
                        chr_mat_2 = normalization(chr_mat_2)

                        chr_mat_3 = cool_3.matrix(
                            balance=BALANCE).fetch(fetch)
                        chr_mat_3 = normalization(chr_mat_3)

                        counter = [1]

                        counter = generate_patch(chr_mat_1, chr_mat_2, chr_mat_3,
                                                 organism, sample, resolution, chromosome, ds_filename, sub_sample, counter)


if __name__ == "__main__":
    try:
        generate_ds(ORGANISMS, ORGANISM_SAMPLES, FILENAME_SET, "inference")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
