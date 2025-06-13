import os
import numpy as np
from PIL import Image
import cooler as cool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import scipy.stats as st
from scipy.ndimage import convolve
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter


root_path = f"/home/mohit/Documents/project/interpolation/data"
out_root_path = f"/home/mohit/Documents/project/interpolation/data/frame/norm/"
resolutions = [10000]
balance = False
patch_sizes = [64, 128, 256]
cmap = "YlOrRd"

# train_organisms = ["human"]
# train_samples = [["atrial", "verticular", "hela_s3"]]
# train_filenames = [[[["a_2d_4DNFIWUZLEWY",
#                     "a_4d_4DNFI7PCOZ9I",
#                      "a_6d_4DNFIWZYSI2D"]],
#                    [["v_2d_4DNFI1P2EP7L",
#                     "v_4d_4DNFI7C5YXNX",
#                      "v_6d_4DNFI8I2WYXS"]],
#                    [["0_5h_4DNFIZZ77KD2",
#                     "1h_4DNFIOLO226X",
#                      "1_5h_4DNFIJMS2ODT"],
#                    ["1_75h_4DNFI49F3LJ4",
#                     "2h_4DNFI65MQOIJ",
#                     "2_25h_4DNFIM4KEPRD"],
#                    ["2_5h_4DNFIIXBIZFC",
#                     "2_75h_4DNFIWDOOBVE",
#                     "3h_4DNFIDT9EB5M"],
#                    ["3_5h_4DNFIEQHTV1R",
#                     "4h_4DNFIFW7GA64",
#                     "4_5h_4DNFIXGXD67I"],
#                    ["5h_4DNFIA7GB1NB",
#                     "6h_4DNFIVOJGWNP",
#                     "7h_4DNFIW22BNB5"],
#                    ["8h_4DNFIIFBC8WN",
#                     "9h_4DNFI9ZBEBJH",
#                     "10h_4DNFID4SLU53"]]]]
# test_organisms = ["human"]
# test_samples = [["atrial", "hela_s3"]]
# test_filenames = [[[["a_2d_4DNFIWUZLEWY",
#                    "a_4d_4DNFI7PCOZ9I",
#                     "a_6d_4DNFIWZYSI2D"]],
#                   [["3h_4DNFIDT9EB5M",
#                    "3_25h_4DNFIX2VUNV8",
#                     "3_5h_4DNFIEQHTV1R"],
#                   ["10h_4DNFID4SLU53",
#                    "11h_4DNFIODI1NUJ",
#                    "12h_4DNFIJL26LFN"]]]]


train_organisms = ["human"]
train_samples = [["verticular", "hela_s3"]]
train_filenames = [[[["v_2d_4DNFI1P2EP7L",
                      "v_4d_4DNFI7C5YXNX",
                     "v_6d_4DNFI8I2WYXS"]],
                   [["3_5h_4DNFIEQHTV1R",
                     "4h_4DNFIFW7GA64",
                     "4_5h_4DNFIXGXD67I"]]]]

test_organisms = ["human"]
test_samples = [["atrial"]]
test_filenames = [[[["a_2d_4DNFIWUZLEWY",
                   "a_4d_4DNFI7PCOZ9I",
                    "a_6d_4DNFIWZYSI2D"]]]]


def save_img(chr_mat, r, c, patch, path, img_name):
    submatrix = chr_mat[r:r+patch, c:c+patch]
    submatrix = submatrix.astype(np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)
    # colormap = cm.get_cmap(cmap)
    # colored_img = colormap(submatrix)
    # colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
    # img = Image.fromarray(colored_img)
    # img.save(f"{path}/{img_name}.jpg")


def generate_patch(mat_0, mat_y, mat_1, organism, sample, resolution, chromosome, ds_filename):
    for patch in patch_sizes:
        ds_file = f"{out_root_path}/{patch}/{ds_filename}.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        # path = f"{out_root_path}/{patch}/{ds_filename}/{organism}/{sample}/{str(resolution)}/{chromosome}"
        # os.makedirs(path, exist_ok=True)
        # np.save(f"{path}/img0.npy", mat_0)
        # np.save(f"{path}/img1.npy", mat_y)
        # np.save(f"{path}/img2.npy", mat_1)

        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating {ds_filename} patches({patch}X{patch}) for {organism} > {sample} > {resolution} > chr{chromosome} ...")
            _count = 1
            row, col = mat_0.shape
            r = 0
            c = 0
            while (r < row):
                while (c < col):
                    folder = f"{_count:06d}"
                    os.makedirs(
                        f"{out_root_path}/{patch}/{ds_filename}/{organism}/{sample}/{str(resolution)}/{chromosome}/{folder}", exist_ok=True)
                    path = f"{ds_filename}/{organism}/{sample}/{str(resolution)}/{chromosome}/{folder}"
                    file.write(path+"\n")

                    path = f"{out_root_path}/{patch}/{ds_filename}/{organism}/{sample}/{str(resolution)}/{chromosome}/{folder}"

                    save_img(mat_0, r, c, patch, path, "img0")
                    save_img(mat_y, r, c, patch, path, "img1")
                    save_img(mat_1, r, c, patch, path, "img2")

                    _count += 1
                    c += patch
                r += patch

# Step 1: Define a custom 2D Gaussian kernel generator


def gkern(kernlen=13, nsig=4):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval/2., nsig + interval/2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

# Step 2: Apply manual Gaussian filter (slower)


def custom_gaussian_filter(matrix, sigma=4, size=13):
    kernel = gkern(size, nsig=sigma)
    result = convolve(matrix, kernel, mode='reflect')
    return result

# Step 3: Faster version using scipy


def fast_gaussian_filter(matrix, sigma=4):
    matrix_gpu = cp.asarray(matrix)  # Transfer to GPU
    result_gpu = gaussian_filter(matrix_gpu, sigma=sigma, mode='reflect')
    return cp.asnumpy(result_gpu)  # Transfer back to CPU


def apply_gaussian_filter(matrix):
    smoothed_fast = fast_gaussian_filter(matrix, sigma=4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(matrix, cmap='Reds')
    axes[0].set_title("Original Matrix")

    axes[1].imshow(smoothed_fast, cmap='Reds')
    axes[1].set_title("Smoothed (CuPy Fast)")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(
        "/home/mohit/Documents/project/interpolation/film/hic_data_preprocessor/comparison.png")


def normalization(matrix):
    np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.log1p(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    norm_matrix = (matrix - min_val) / (max_val - min_val)
    norm_matrix = fast_gaussian_filter(norm_matrix)
    return norm_matrix


def generate_ds(organisms, samples, filename_set, ds_filename):
    for organism, org_samples, org_filenames in zip(organisms, samples, filename_set):
        for sample, sample_filenames in zip(org_samples, org_filenames):
            for resolution in resolutions:
                for filenames in sample_filenames:

                    cool_0 = cool.Cooler(
                        f"{root_path}/{organism}/{sample}/{filenames[0]}_{resolution}.cool")
                    cool_y = cool.Cooler(
                        f"{root_path}/{organism}/{sample}/{filenames[1]}_{resolution}.cool")
                    cool_1 = cool.Cooler(
                        f"{root_path}/{organism}/{sample}/{filenames[2]}_{resolution}.cool")

                    for chromosome in cool_0.chromnames:
                        chr_mat_0 = cool_0.matrix(balance=balance).fetch(
                            chromosome)
                        chr_mat_0 = normalization(chr_mat_0)
                        chr_mat_y = cool_y.matrix(balance=balance).fetch(
                            chromosome)
                        chr_mat_y = normalization(chr_mat_y)
                        chr_mat_1 = cool_1.matrix(balance=balance).fetch(
                            chromosome)
                        chr_mat_1 = normalization(chr_mat_1)
                        generate_patch(chr_mat_0, chr_mat_y, chr_mat_1,
                                       organism, sample, resolution, chromosome, ds_filename)


generate_ds(train_organisms, train_samples, train_filenames, "train")
generate_ds(test_organisms, test_samples, test_filenames, "test")
