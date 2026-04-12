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
PATCHES = [512, 256, 128]
_CMAP = "Reds"
PATCH_OVERLAP_RATIO = 0.2
WRITE_BUFFER_LINES = 4096
RESOLUTION_STR = ["5000", "10000", "25000"]

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
    submatrix = np.asarray(chr_mat[r:r+patch, c:c+patch], dtype=np.float32)
    np.save(f"{path}/{img_name}.npy", submatrix)


def generate_patch(mat_0, mat_y, mat_1, organism, sample, resolution, chromosome, sub_sample, counter, output_root_path, ds_file_handles, created_dirs):
    for patch, i in zip(PATCHES, range(0, len(counter))):
        ds_handle = ds_file_handles[patch]
        print(
            f"[INFO] generating patches({patch}X{patch}) for {organism} > {sample} > {sub_sample} > {resolution} > chr{chromosome}")
        row, col = mat_y.shape
        bin_inc = int(patch * (1 - PATCH_OVERLAP_RATIO))
        r = 0
        c = 0
        line_buffer = []
        while (r + patch <= row and c + patch <= col):
            folder = f"{counter[i]:08d}"
            rel_path = f"{organism}/{sample}/{sub_sample}/{str(resolution)}/{chromosome}/{folder}"
            line_buffer.append(rel_path + "\n")
            if len(line_buffer) >= WRITE_BUFFER_LINES:
                ds_handle.writelines(line_buffer)
                line_buffer.clear()
            abs_path = f"{output_root_path}/{resolution}/{patch}/{rel_path}"
            if abs_path not in created_dirs:
                os.makedirs(abs_path, exist_ok=True)
                created_dirs.add(abs_path)
            save_img(mat_0, r, c, patch, abs_path, "img1")
            save_img(mat_y, r, c, patch, abs_path, "img2")
            save_img(mat_1, r, c, patch, abs_path, "img3")
            counter[i] += 1
            c += bin_inc
            r += bin_inc

        if line_buffer:
            ds_handle.writelines(line_buffer)

    return counter


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

        # Keep one open index file per patch for this resolution.
        ds_file_handles = {}
        created_dirs = set()
        for patch in PATCHES:
            patch_output_path = f"{resolution_output_path}/{patch}"
            os.makedirs(patch_output_path, exist_ok=True)
            ds_file = f"{patch_output_path}/dataset_dict.txt"
            ds_file_handles[patch] = open(ds_file, "w")

        try:
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

                            mat_0_selector = cool_0.matrix(
                                balance=BALANCE_COOL)
                            mat_y_selector = cool_y.matrix(
                                balance=BALANCE_COOL)
                            mat_1_selector = cool_1.matrix(
                                balance=BALANCE_COOL)

                            for chromosome, chr_size in zip(cool_y.chromnames, cool_y.chromsizes):
                                fetch = f"{chromosome}:0-{chr_size}"
                                chr_mat_0 = mat_0_selector.fetch(fetch)
                                chr_mat_y = mat_y_selector.fetch(fetch)
                                chr_mat_1 = mat_1_selector.fetch(fetch)
                                counter = generate_patch(chr_mat_0, chr_mat_y, chr_mat_1,
                                                         organism, sample, resolution, chromosome, sub_sample, counter,
                                                         output_root_path=output_root_path,
                                                         ds_file_handles=ds_file_handles,
                                                         created_dirs=created_dirs)
        finally:
            for ds_handle in ds_file_handles.values():
                ds_handle.close()


if __name__ == "__main__":
    try:
        output_root_path = f"{ROOT_PATH}/triplates/human"
        generate_ds(ORGANISMS, SAMPLES, SUBSAMPLES, FILENAME_LIST,
                    output_root_path=output_root_path, gf=True, log=False, clip=False)
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
