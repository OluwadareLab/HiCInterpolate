import os
import numpy as np
import cooler as cool
from scipy.ndimage import gaussian_filter as sp_gf
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as cp_gf
import matplotlib.pyplot as plt

ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets"
RESOLUTIONS = [5000, 10000, 25000]
BALANCE_COOL = True
PATCHES = [64]
_CMAP = "Reds"
_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99
PATCH_OVERLAP_RATIO = 0.2

RESOLUTION_STR = ["5000", "10000", "25000"]
ALPHA = [0.5, 0.75, 1.0, 1.25]

ORGANISMS = ["human"]
SAMPLES = [
    [
        "dmso",
        "hct116",
        "hela_s3"
    ]
]

SUBSAMPLES = [
    [
        ["control"],
        ["2"],
        ["r1", "r2", "r3"]
    ]
]

FILENAME_LIST = [
    [
        [
            [
                ["4DNFIP9EJSOM_dmso_control_0m",
                 "4DNFI7T93SHL_dmso_control_30m",
                 "4DNFICF2Z2TG_dmso_control_60m"],

                ["4DNFI7T93SHL_dmso_control_30m",
                 "4DNFICF2Z2TG_dmso_control_60m",
                 "4DNFILL624WG_dmso_control_90m"],

                ["4DNFICF2Z2TG_dmso_control_60m",
                 "4DNFILL624WG_dmso_control_90m",
                 "4DNFIC4GB8UM_dmso_control_120m"]
            ],
            [
                ["4DNFIAAH19VM_hct116_2_20m",
                 "4DNFI7QUSU5J_hct116_2_40m",
                 "4DNFIXEB4UZO_hct116_2_60m"],

                ["4DNFI5IZNXIO_hct116_2_no_transcription_360m_20m",
                 "4DNFIZK7W8GZ_hct116_2_no_transcription_360m_40m",
                 "4DNFISRP84FE_hct116_2_no_transcription_360m_60m"],

                ["4DNFII16KXA7_hct116_2_no_transcription_60m_20m",
                 "4DNFIMIMLMD3_hct116_2_no_transcription_60m_40m",
                 "4DNFI2LY7B73_hct116_2_no_transcription_60m_60m"],

                ["4DNFITUPI4HA_hct116_2_no_atp_120m_20m",
                 "4DNFIM7Q2FQQ_hct116_2_no_atp_120m_40m",
                 "4DNFISATK9PF_hct116_2_no_atp_120m_60m"],

                ["4DNFIVC8OQPG_hct116_2_no_atp_30m_20m",
                 "4DNFI44JLUSL_hct116_2_no_atp_30m_40m",
                 "4DNFIBED48O1_hct116_2_no_atp_30m_60m"],

                ["4DNFIDD9IF9T_hct116_2_no_replication_20m",
                 "4DNFIQWWATGK_hct116_2_no_replication_40m",
                 "4DNFI3NTD7B3_hct116_2_no_replication_60m"]
            ], [
                ["4DNFIZZ77KD2_hela_s3_r1_30m",
                 "4DNFIOLO226X_hela_s3_r1_60m",
                 "4DNFIJMS2ODT_hela_s3_r1_90m"],

                ["4DNFIJMS2ODT_hela_s3_r1_90m",
                 "4DNFI49F3LJ4_hela_s3_r1_105m",
                 "4DNFI65MQOIJ_hela_s3_r1_120m"],

                ["4DNFI49F3LJ4_hela_s3_r1_105m",
                 "4DNFI65MQOIJ_hela_s3_r1_120m",
                 "4DNFIM4KEPRD_hela_s3_r1_135m"],

                ["4DNFI65MQOIJ_hela_s3_r1_120m",
                 "4DNFIM4KEPRD_hela_s3_r1_135m",
                 "4DNFIIXBIZFC_hela_s3_r1_150m"],

                ["4DNFIM4KEPRD_hela_s3_r1_135m",
                 "4DNFIIXBIZFC_hela_s3_r1_150m",
                 "4DNFIWDOOBVE_hela_s3_r1_165m"],

                ["4DNFIIXBIZFC_hela_s3_r1_150m",
                 "4DNFIWDOOBVE_hela_s3_r1_165m",
                 "4DNFIDT9EB5M_hela_s3_r1_180m"],

                ["4DNFIWDOOBVE_hela_s3_r1_165m",
                 "4DNFIDT9EB5M_hela_s3_r1_180m",
                 "4DNFIX2VUNV8_hela_s3_r1_195m"],

                ["4DNFIDT9EB5M_hela_s3_r1_180m",
                 "4DNFIX2VUNV8_hela_s3_r1_195m",
                 "4DNFIEQHTV1R_hela_s3_r1_210m"],

                ["4DNFIEQHTV1R_hela_s3_r1_210m",
                 "4DNFIFW7GA64_hela_s3_r1_240m",
                 "4DNFIXGXD67I_hela_s3_r1_270m"],

                ["4DNFIFW7GA64_hela_s3_r1_240m",
                 "4DNFIXGXD67I_hela_s3_r1_270m",
                 "4DNFIA7GB1NB_hela_s3_r1_300m"]
            ], [
                ["4DNFIX6ZXCA8_hela_s3_r2_30m",
                 "4DNFIEVR81FS_hela_s3_r2_60m",
                 "4DNFIAUI6BBI_hela_s3_r2_90m"],

                ["4DNFIEVR81FS_hela_s3_r2_60m",
                 "4DNFIAUI6BBI_hela_s3_r2_90m",
                 "4DNFIAFEE9G2_hela_s3_r2_120m"],

                ["4DNFIAUI6BBI_hela_s3_r2_90m",
                 "4DNFIAFEE9G2_hela_s3_r2_120m",
                 "4DNFIPZBEXCP_hela_s3_r2_150m"],

                ["4DNFIAFEE9G2_hela_s3_r2_120m",
                 "4DNFIPZBEXCP_hela_s3_r2_150m",
                 "4DNFIWPKRZGU_hela_s3_r2_180m"],

                ["4DNFIPZBEXCP_hela_s3_r2_150m",
                 "4DNFIWPKRZGU_hela_s3_r2_180m",
                 "4DNFIMD9QNDX_hela_s3_r2_210m"],

                ["4DNFIWPKRZGU_hela_s3_r2_180m",
                 "4DNFIMD9QNDX_hela_s3_r2_210m",
                 "4DNFIATA1HD5_hela_s3_r2_240m"],

                ["4DNFIMD9QNDX_hela_s3_r2_210m",
                 "4DNFIATA1HD5_hela_s3_r2_240m",
                 "4DNFIH9U4I7I_hela_s3_r2_270m"],

                ["4DNFIATA1HD5_hela_s3_r2_240m",
                 "4DNFIH9U4I7I_hela_s3_r2_270m",
                 "4DNFIZ95S6TR_hela_s3_r2_300m"]
            ], [
                ["4DNFICFZGFAV_hela_s3_r3_30m",
                 "4DNFIQXCZVVA_hela_s3_r3_60m",
                 "4DNFIB6PJFJ3_hela_s3_r3_90m"],

                ["4DNFIB6PJFJ3_hela_s3_r3_90m",
                 "4DNFIX97731O_hela_s3_r3_105m",
                 "4DNFIYQYZOTO_hela_s3_r3_120m"],

                ["4DNFIX97731O_hela_s3_r3_105m",
                 "4DNFIYQYZOTO_hela_s3_r3_120m",
                 "4DNFIPXU7V25_hela_s3_r3_135m"],

                ["4DNFIYQYZOTO_hela_s3_r3_120m",
                 "4DNFIPXU7V25_hela_s3_r3_135m",
                 "4DNFIL39PR76_hela_s3_r3_150m"],

                ["4DNFIPXU7V25_hela_s3_r3_135m",
                 "4DNFIL39PR76_hela_s3_r3_150m",
                 "4DNFIYLJ3R3B_hela_s3_r3_165m"],

                ["4DNFIL39PR76_hela_s3_r3_150m",
                 "4DNFIYLJ3R3B_hela_s3_r3_165m",
                 "4DNFIL51WBN6_hela_s3_r3_180m"],

                ["4DNFIYLJ3R3B_hela_s3_r3_165m",
                 "4DNFIL51WBN6_hela_s3_r3_180m",
                 "4DNFI6SFPUDA_hela_s3_r3_195m"],

                ["4DNFIL51WBN6_hela_s3_r3_180m",
                 "4DNFI6SFPUDA_hela_s3_r3_195m",
                 "4DNFI2KM22QR_hela_s3_r3_210m"],

                ["4DNFI2KM22QR_hela_s3_r3_210m",
                 "4DNFIVF8Q45U_hela_s3_r3_240m",
                 "4DNFI2RN3WFP_hela_s3_r3_270m"],

                ["4DNFIVF8Q45U_hela_s3_r3_240m",
                 "4DNFI2RN3WFP_hela_s3_r3_270m",
                 "4DNFI4TJTL7A_hela_s3_r3_300m"]
            ]
        ]
    ]
]


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


def generate_patch(mat_0, mat_y, mat_1, organism, sample, resolution, chromosome, sub_sample, counter, output_root_path):
    for patch, i in zip(PATCHES, range(0, len(counter))):
        ds_file = f"{output_root_path}/{resolution}/{patch}/dataset_dict.txt"
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)
        with open(ds_file, "a") as file:
            print(
                f"[INFO] generating patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {resolution} > chr{chromosome}")
            row, col = mat_y.shape
            bin_inc = int(patch*(1-PATCH_OVERLAP_RATIO))
            window = [0]
            for win in window:
                r = win
                c = 0
                while (r+patch <= row and c+patch <= col):
                    if r < 0 or c < 0:
                        c += bin_inc
                        r += bin_inc
                        continue
                    folder = f"{counter[i]:08d}"
                    path = f"{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}/{folder}"
                    file.write(path+"\n")
                    path = f"{output_root_path}/{resolution}/{patch}/{path}"
                    os.makedirs(path, exist_ok=True)
                    save_img(mat_0, r, c, patch, path, "img1")
                    save_img(mat_y, r, c, patch, path, "img2")
                    save_img(mat_1, r, c, patch, path, "img3")
                    counter[i] += 1
                    c += bin_inc
                    r += bin_inc

    return counter


def get_cp_gf(matrix, sigma=0.75):
    try:
        with cp.cuda.Device(0):
            matrix_gpu = cp.asarray(matrix)
            result_gpu = cp_gf(matrix_gpu, sigma=sigma, mode='nearest')
            result_cpu = cp.asnumpy(result_gpu)
            del matrix_gpu, result_gpu

            cp._default_memory_pool.free_all_blocks()
            return result_cpu
    except cp.cuda.memory.OutOfMemoryError:
        print("[ERROR] CuPy ran out of GPU memory.")
        raise cp.cuda.memory.OutOfMemoryError


def raw_matrix(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    return matrix


def gf_norm(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    gf_matrix = sp_gf(matrix, 0.75)
    _min = np.min(gf_matrix)
    _max = np.max(gf_matrix)
    denom = _max - _min
    if denom <= _EPSILON:
        mm_matrix = np.full_like(gf_matrix, _EPSILON)
        return mm_matrix
    mm_matrix = (gf_matrix - _min)/denom
    mm_matrix[mm_matrix == 0] = _EPSILON
    return mm_matrix


def min_max_norm(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    _min = np.min(matrix)
    _max = np.max(matrix)
    denom = _max - _min
    if denom <= _EPSILON:
        mm_matrix = np.full_like(matrix, _EPSILON)
        return mm_matrix
    mm_matrix = (matrix - _min)/denom
    mm_matrix[mm_matrix == 0] = _EPSILON
    return mm_matrix


def log_clip(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    log_matrix = np.log1p(matrix)
    percentile_val = np.percentile(log_matrix, CLIPPING_PERCENTILE)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)

    return clip_matrix


def rev_log_clip_min_max(matrix):
    mat = np.expm1(matrix)
    log_matrix = np.log1p(mat)
    return log_matrix


def log_clip_min_max(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    log_matrix = np.log1p(matrix)
    percentile_val = np.percentile(log_matrix, CLIPPING_PERCENTILE)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)
    norm_matrix = clip_matrix / percentile_val

    return norm_matrix


def normalization(matrix):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    log_matrix = np.log1p(matrix)
    percentile_val = np.percentile(log_matrix, CLIPPING_PERCENTILE)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)
    norm_matrix = clip_matrix / percentile_val

    return norm_matrix


def get_norm_mat(matrix, gf: bool = False, log: bool = False, clip: bool = False):
    mat = np.nan_to_num(matrix, nan=_EPSILON, posinf=_EPSILON, neginf=_EPSILON)
    if gf:
        mat = sp_gf(mat, 1.0)
    if log:
        mat = np.log1p(mat)
    if clip:
        percentile_val = np.percentile(mat, CLIPPING_PERCENTILE)
        mat = np.clip(mat, _EPSILON, percentile_val)

    _min = np.min(mat)
    _max = np.max(mat)
    denom = _max - _min
    if denom <= _EPSILON:
        return np.full_like(mat, _EPSILON)
    mat = (mat - _min)/denom
    mat[mat == 0] = _EPSILON

    return mat


def _unwrap_singleton_lists(items):
    result = items
    while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    return result


def _sample_matches_filename(sample, filename):
    sample_lower = sample.lower()
    filename_lower = filename.lower()

    if sample_lower in filename_lower:
        return True

    # Handle known naming mismatch in input lists: dmsol vs dmso.
    aliases = {
        "dmsol": "dmso",
        "dmso": "dmsol",
    }
    alias = aliases.get(sample_lower)
    return alias in filename_lower if alias else False


def _normalize_subsample_group(group):
    if isinstance(group, str):
        return [group]

    if not isinstance(group, list):
        return []

    result = []
    stack = list(group)
    while stack:
        item = stack.pop(0)
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, list):
            stack = item + stack
    return result


def _sample_subsample_map(org_samples, org_subsamples, triplet_groups):
    active_samples = []
    for group in triplet_groups:
        if not group or not group[0]:
            continue
        sample_name = None
        first_filename = group[0][0]
        for sample in org_samples:
            if _sample_matches_filename(sample, first_filename):
                sample_name = sample
                break
        if sample_name and sample_name not in active_samples:
            active_samples.append(sample_name)

    sample_to_subsamples = {}
    if len(org_subsamples) == len(org_samples):
        for idx, sample in enumerate(org_samples):
            sample_to_subsamples[sample] = _normalize_subsample_group(
                org_subsamples[idx])
        return sample_to_subsamples

    if len(org_subsamples) == len(active_samples):
        for idx, sample in enumerate(active_samples):
            sample_to_subsamples[sample] = _normalize_subsample_group(
                org_subsamples[idx])
        return sample_to_subsamples

    raise ValueError(
        "SUBSAMPLES layout does not match SAMPLES or detected active samples. "
        f"Got {len(org_subsamples)} subsample groups, {len(org_samples)} samples, "
        f"and {len(active_samples)} active samples in filename list."
    )


def _subsample_matches_filename(sub_sample, filename):
    key = sub_sample.lower()
    text = filename.lower()
    return (
        f"_{key}_" in text
        or text.endswith(f"_{key}")
        or text.startswith(f"{key}_")
    )


def _find_subsample_for_group(group, sample, sample_to_subsamples):
    subsamples = sample_to_subsamples.get(sample, [])
    if len(subsamples) == 1:
        return subsamples[0]

    if not group or not group[0]:
        return None

    first_filename = group[0][0]
    matches = [sub for sub in subsamples if _subsample_matches_filename(
        sub, first_filename)]
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return max(matches, key=len)

    return None


def _group_triplets_by_sample_subsample(org_samples, org_subsamples, triplet_groups):
    sample_to_subsamples = _sample_subsample_map(
        org_samples, org_subsamples, triplet_groups)
    grouped_triplets = {}

    for group in triplet_groups:
        if not group or not group[0]:
            continue

        first_filename = group[0][0]
        sample = None
        for sample_name in org_samples:
            if _sample_matches_filename(sample_name, first_filename):
                sample = sample_name
                break

        if sample is None:
            raise ValueError(
                f"Unable to match filename group to a sample: {first_filename}"
            )

        sub_sample = _find_subsample_for_group(
            group, sample, sample_to_subsamples)
        if sub_sample is None:
            raise ValueError(
                "Unable to match filename group to a subsample for sample "
                f"'{sample}': {first_filename}"
            )

        key = (sample, sub_sample)
        if key not in grouped_triplets:
            grouped_triplets[key] = []
        grouped_triplets[key].extend(group)

    return sample_to_subsamples, grouped_triplets


def generate_ds(organisms, samples, subsamples, filename_list, output_root_path: str, gf: bool, log: bool, clip: bool):
    os.makedirs(output_root_path, exist_ok=True)

    for resolution in RESOLUTIONS:
        resolution_output_path = f"{output_root_path}/{resolution}"
        os.makedirs(resolution_output_path, exist_ok=True)

        # Start each run with a fresh dataset index file per patch size and resolution.
        for patch in PATCHES:
            patch_output_path = f"{resolution_output_path}/{patch}"
            os.makedirs(patch_output_path, exist_ok=True)
            ds_file = f"{patch_output_path}/dataset_dict.txt"
            with open(ds_file, "w"):
                pass

        # Keep counters unique across all datasets/chromosomes for this resolution,
        # then reset automatically when moving to the next resolution.
        counter = [1] * len(PATCHES)
        for organism, org_samples, org_subsamples, org_filenames in zip(organisms, samples, subsamples, filename_list):
            triplet_groups = _unwrap_singleton_lists(org_filenames)

            sample_to_subsamples, grouped_triplets = _group_triplets_by_sample_subsample(
                org_samples, org_subsamples, triplet_groups
            )

            for sample in org_samples:
                for sub_sample in sample_to_subsamples.get(sample, []):
                    sample_triplets = grouped_triplets.get(
                        (sample, sub_sample), [])
                    for filenames in sample_triplets:
                        if len(filenames) < 3:
                            print(
                                "[WARN] Skipping invalid triplet (expected 3 files): "
                                f"{filenames}"
                            )
                            continue

                        cool_0 = cool.Cooler(
                            f"{ROOT_PATH}/{organism}/{sample}/{sub_sample}/{filenames[0]}_{resolution}_KR.cool")
                        cool_y = cool.Cooler(
                            f"{ROOT_PATH}/{organism}/{sample}/{sub_sample}/{filenames[1]}_{resolution}_KR.cool")
                        cool_1 = cool.Cooler(
                            f"{ROOT_PATH}/{organism}/{sample}/{sub_sample}/{filenames[2]}_{resolution}_KR.cool")

                        for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                            fetch = f"{chromosome}:{0}-{chr_size}"
                            chr_mat_0 = cool_0.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            chr_mat_0 = get_norm_mat(
                                matrix=chr_mat_0, gf=gf, log=log, clip=clip)
                            chr_mat_y = cool_y.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            chr_mat_y = get_norm_mat(
                                matrix=chr_mat_y, gf=gf, log=log, clip=clip)
                            chr_mat_1 = cool_1.matrix(
                                balance=BALANCE_COOL).fetch(fetch)
                            chr_mat_1 = get_norm_mat(
                                matrix=chr_mat_1, gf=gf, log=log, clip=clip)
                            counter = generate_patch(chr_mat_0, chr_mat_y, chr_mat_1,
                                                     organism, sample, resolution, chromosome, sub_sample, counter, output_root_path=output_root_path)


if __name__ == "__main__":
    try:
        output_root_path = f"{ROOT_PATH}/triplates/human/alpha_75"
        generate_ds(ORGANISMS, SAMPLES, SUBSAMPLES, FILENAME_LIST,
                    output_root_path=output_root_path, gf=True, log=False, clip=False)
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
