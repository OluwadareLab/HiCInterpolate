import os
import numpy as np
import cooler as cool
from scipy.ndimage import gaussian_filter
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

root_path = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data"
out_root_path = f"{root_path}/triplets/kr_log_clip_norm_diag"
resolutions = [10000]
balance = True
patch_sizes = [256]
# CMAP_ = "YlOrRd"
CMAP_ = "Reds"
EPSILON = 1e-8
PERCENTILE = 99.99
OVERLAP_RATIO = 0.2

train_organisms = ["human"]
train_samples = [["atrial", "verticular", "hela_s3"]]
train_filenames = [[[["a_2d_4DNFIWUZLEWY",
                    "a_4d_4DNFI7PCOZ9I",
                     "a_6d_4DNFIWZYSI2D"]],
                   [["v_2d_4DNFI1P2EP7L",
                    "v_4d_4DNFI7C5YXNX",
                     "v_6d_4DNFI8I2WYXS"]],
                   [["0_5h_4DNFIZZ77KD2",
                    "1h_4DNFIOLO226X",
                     "1_5h_4DNFIJMS2ODT"],
                   ["1_75h_4DNFI49F3LJ4",
                    "2h_4DNFI65MQOIJ",
                    "2_25h_4DNFIM4KEPRD"],
                   ["2_5h_4DNFIIXBIZFC",
                    "2_75h_4DNFIWDOOBVE",
                    "3h_4DNFIDT9EB5M"],
                   ["3_5h_4DNFIEQHTV1R",
                    "4h_4DNFIFW7GA64",
                    "4_5h_4DNFIXGXD67I"],
                   ["5h_4DNFIA7GB1NB",
                    "6h_4DNFIVOJGWNP",
                    "7h_4DNFIW22BNB5"],
                   ["8h_4DNFIIFBC8WN",
                    "9h_4DNFI9ZBEBJH",
                    "10h_4DNFID4SLU53"]]]]

test_organisms = ["human"]
test_samples = [["hela_s3"]]
test_filenames = [[[["3h_4DNFIDT9EB5M",
                   "3_25h_4DNFIX2VUNV8",
                    "3_5h_4DNFIEQHTV1R"],
                  ["10h_4DNFID4SLU53",
                   "11h_4DNFIODI1NUJ",
                   "12h_4DNFIJL26LFN"]]]]

# train_organisms = ["human"]
# train_samples = [["verticular", "hela_s3"]]
# train_filenames = [[[["v_2d_4DNFI1P2EP7L",
#                       "v_4d_4DNFI7C5YXNX",
#                      "v_6d_4DNFI8I2WYXS"]],
#                    [["3_5h_4DNFIEQHTV1R",
#                      "4h_4DNFIFW7GA64",
#                      "4_5h_4DNFIXGXD67I"]]]]

# test_organisms = ["human"]
# test_samples = [["atrial"]]
# test_filenames = [[[["a_2d_4DNFIWUZLEWY",
#                    "a_4d_4DNFI7PCOZ9I",
#                     "a_6d_4DNFIWZYSI2D"]]]]


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch]
    submatrix = submatrix.astype(np.float32)
    # plot_hic_map(submatrix, f"{patch}_{img_name}")
    np.save(f"{path}/{img_name}.npy", submatrix)


def generate_patch(mat_0, mat_y, mat_1, organism, sample, resolution, chromosome, ds_filename, sub_sample, counter):
    for patch, i in zip(patch_sizes, range(0, len(counter))):
        ds_file = f"{out_root_path}/{patch}/{ds_filename}.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating {ds_filename} patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {resolution} > chr{chromosome}")
            row, col = mat_y.shape
            bin_inc = int(patch*(1-OVERLAP_RATIO))
            win_start = patch//2
            # col_start = [0, -bin_inc, bin_inc]
            # window = [-win_start, 0, win_start]
            window = [0]
            for win in window:
                r = win
                c = 0
                while (r+patch <= row and c+patch <= col):
                    if r < 0 or c < 0:
                        c += bin_inc
                        r += bin_inc
                        continue
                    folder = f"{counter[i]:06d}"
                    path = f"{ds_filename}/{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}/{folder}"
                    file.write(path+"\n")
                    path = f"{out_root_path}/{patch}/{path}"
                    os.makedirs(path, exist_ok=True)
                    save_img(mat_0, r, c, patch, path, "img1")
                    save_img(mat_y, r, c, patch, path, "img2")
                    save_img(mat_1, r, c, patch, path, "img3")
                    counter[i] += 1
                    c += bin_inc
                    r += bin_inc

            # col_start = [0]
            # for cs in col_start:
            #     c = cs
            #     r = 0
            #     # print(f"Diag:", end=' ')
            #     while (r+patch <= row and c+patch <= col):
            #         if c >= 0:
            #             folder = f"{counter[i]:06d}"
            #             path = f"{ds_filename}/{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}/{folder}"
            #             file.write(path+"\n")
            #             path = f"{out_root_path}/{patch}/{path}"
            #             os.makedirs(path, exist_ok=True)
            #             # print(f"({r}, {c})", end=' ')
            #             save_img(mat_0, r, c, patch, path, "img1")
            #             save_img(mat_y, r, c, patch, path, "img2")
            #             save_img(mat_1, r, c, patch, path, "img3")
            #             counter[i] += 1

            #         c += bin_inc
            #         r += bin_inc

                # c = np.random.randint(cs+(patch*2), col)
                # r = 0
                # rand_c = 0
                # max_rand_c = int((col//patch) * 0.25)
                # # print(f"\nRand:", end=' ')
                # while (r+patch <= row and c+patch <= col and rand_c < max_rand_c):
                #     if c >= 0:
                #         folder = f"{counter[i]:06d}"
                #         path = f"{ds_filename}/{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}/{folder}"
                #         file.write(path+"\n")
                #         path = f"{out_root_path}/{patch}/{path}"
                #         os.makedirs(path, exist_ok=True)
                #         # print(f"({r}, {c})", end=' ')
                #         save_img(mat_0, r, c, patch, path, "img1")
                #         save_img(mat_y, r, c, patch, path, "img2")
                #         save_img(mat_1, r, c, patch, path, "img3")
                #         counter[i] += 1
                #         rand_c += 1

                #     c += patch
                #     r += patch

    return counter


def fast_gaussian_filter(matrix, sigma=4):
    try:
        with cp.cuda.Device(0):
            matrix_gpu = cp.asarray(matrix)
            result_gpu = gaussian_filter(
                matrix_gpu, sigma=sigma, mode='reflect')
            result_cpu = cp.asnumpy(result_gpu)
            del matrix_gpu, result_gpu
            cp._default_memory_pool.free_all_blocks()
            return result_cpu
    except cp.cuda.memory.OutOfMemoryError:
        print("[ERROR] CuPy ran out of GPU memory.")
        raise cp.cuda.memory.OutOfMemoryError


def plot_hic_map(matrix, filename):
    plt.imshow(matrix, cmap=CMAP_)
    plt.title(f"{filename}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, format='png')
    plt.close()


def normalization(matrix):
    matrix = np.nan_to_num(matrix, nan=EPSILON, posinf=EPSILON, neginf=EPSILON)
    log_matrix = np.log1p(matrix+1)
    percentile_val = np.percentile(log_matrix, PERCENTILE)
    clip_matrix = np.clip(log_matrix, EPSILON, percentile_val)
    norm_matrix = clip_matrix / percentile_val

    # plt.figure(figsize=(10, 5))
    # plt.hist(matrix.ravel(),
    #          bins=500, alpha=0.5, label='original')
    # plt.hist(log_matrix.ravel(),
    #          bins=500, alpha=0.5, label='log')
    # plt.hist(clip_matrix.ravel(),
    #          bins=500, alpha=0.5, label='clipped')
    # values = matrix.ravel()
    # lower = np.percentile(values, 0.1)
    # upper = np.percentile(values, 99.9)

    # plt.hist(values, bins=200, alpha=0.5, edgecolor='black')
    # plt.xlim(lower, upper)
    # plt.yscale('log')
    # plt.xlabel('Value')
    # plt.ylabel('Count')
    # plt.title('Hi-C distribution zoomed by percentile')
    # plt.tight_layout()
    # plt.savefig('hist_percentile.png', dpi=300)
    # plt.close()

    # import seaborn as sns
    # sns.kdeplot(values, bw_adjust=0.5)
    # plt.xlim(lower, upper)
    # plt.title('KDE of Hi-C values (zoomed)')
    # plt.tight_layout()
    # plt.savefig('kde_zoom.png', dpi=300)
    # plt.close()

    # counts, bin_edges = np.histogram(values, bins=20)
    # for i in range(len(counts)):
    #     print(f"{bin_edges[i]:8.3f} – {bin_edges[i+1]:8.3f} : {counts[i]}")

    return norm_matrix


def generate_ds(organisms, samples, filename_set, ds_filename):
    for organism, org_samples, org_filenames in zip(organisms, samples, filename_set):
        for sample, sample_filenames in zip(org_samples, org_filenames):
            for resolution in resolutions:
                for filenames in sample_filenames:
                    cool_0 = cool.Cooler(
                        f"{root_path}/{organism}/{sample}/{filenames[0]}_{resolution}_KR.cool")
                    cool_y = cool.Cooler(
                        f"{root_path}/{organism}/{sample}/{filenames[1]}_{resolution}_KR.cool")
                    cool_1 = cool.Cooler(
                        f"{root_path}/{organism}/{sample}/{filenames[2]}_{resolution}_KR.cool")

                    sub_sample = "_".join(name.split(
                        '_')[-1] for name in filenames[:3])
                    for chromosome, chr_size in zip(cool_0.chromnames, cool_0.chromsizes):

                        fetch = f"{chromosome}:{0}-{chr_size}"
                        chr_mat_0 = cool_0.matrix(
                            balance=balance).fetch(fetch)
                        chr_mat_0 = normalization(chr_mat_0)
                        chr_mat_y = cool_y.matrix(
                            balance=balance).fetch(fetch)
                        chr_mat_y = normalization(chr_mat_y)
                        chr_mat_1 = cool_1.matrix(
                            balance=balance).fetch(fetch)
                        chr_mat_1 = normalization(chr_mat_1)

                        counter = [1]
                        counter = generate_patch(chr_mat_0, chr_mat_y, chr_mat_1,
                                                 organism, sample, resolution, chromosome, ds_filename, sub_sample, counter)


if __name__ == "__main__":
    try:
        generate_ds(train_organisms, train_samples, train_filenames, "train")
        generate_ds(test_organisms, test_samples, test_filenames, "test")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
