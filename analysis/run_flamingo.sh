#!/usr/bin/env bash
# FLAMINGO 3D structures (bins 2500-4000) for all run_mustache datasets, then SCC vs y.
# Usage:
#   bash run_flamingo.sh
#   CHROMS="10 11" bash run_flamingo.sh
#   MIN_BINS=2500 MAX_BINS=4000 bash run_flamingo.sh
set -euo pipefail

SCRIPT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate/analysis"
RSCRIPT="${RSCRIPT:-/home/hc0783.unt.ad.unt.edu/.conda/envs/flamingo/bin/Rscript}"
INPUT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/config_25k_64"
OUTPUT_ROOT="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/flamingo_user"

RESOLUTION="${RESOLUTION:-25000}"
MODEL_PATCH="${MODEL_PATCH:-64}"
DOMAIN_RES="${DOMAIN_RES:-1000000}"
MIN_BINS="${MIN_BINS:-2500}"
MAX_BINS="${MAX_BINS:-4000}"
N_THREAD="${N_THREAD:-1}"
METHODS=(${METHODS:-y pred of linear 4dmax})
SCC_CSV="${SCC_CSV:-${OUTPUT_ROOT}/flamingo_scc_${MIN_BINS}_${MAX_BINS}.csv}"

HUMAN_CHROMS=(${CHROMS:-10 11 15 16 20 21})
MOUSE_CHROMS=(${MOUSE_CHROMS:-10 15 19})

HUMAN_TRIPLETS=(
    "dmso|control|dmso_control_30m|dmso_control_60m|dmso_control_90m"
    "dtag|v1|dtag_v1_30m|dtag_v1_60m|dtag_v1_90m"
)

MOUSE_TRIPLETS=(
    "cerebellar_granule_neuron|control|cerebellar_granule_neuron_control_10080m|cerebellar_granule_neuron_control_11520m|cerebellar_granule_neuron_control_12960m"
    "embryo|development|zygote|early2_cell|late2_cell"
    "embryo|development|early2_cell|late2_cell|8cell"
    "embryo|development|late2_cell|8cell|icm"
)

if (( MIN_BINS < 0 || MAX_BINS <= MIN_BINS )); then
    echo "[run_flamingo] need 0 <= MIN_BINS < MAX_BINS (got ${MIN_BINS}, ${MAX_BINS})" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "[run_flamingo] region bins [${MIN_BINS}, ${MAX_BINS}] @ ${RESOLUTION}"
echo "[run_flamingo] methods=${METHODS[*]}"

# Single fixed region [MIN_BINS, MAX_BINS], clipped to matrix size.
build_regions() {
    local n_bins="$1"
    REGIONS_ARR=()
    if (( n_bins <= MIN_BINS )); then
        echo "[run_flamingo] SKIP region: n_bins=${n_bins} <= MIN_BINS=${MIN_BINS}" >&2
        return
    fi
    local end=$MAX_BINS
    if (( end > n_bins )); then
        end=$n_bins
    fi
    if (( end > MIN_BINS )); then
        REGIONS_ARR+=("${MIN_BINS}:${end}")
    fi
}

process_triplet_set() {
    local organism="$1"
    shift
    local -a chroms=("$@")
    local -a triplets=()
    if [[ "$organism" == "human" ]]; then
        triplets=("${HUMAN_TRIPLETS[@]}")
    else
        triplets=("${MOUSE_TRIPLETS[@]}")
    fi

    local entry sample subsample t0 t1 t2 timestamp chromosome
    local sample_tag out_tag prefix y_matrix n_bins method matrix_file
    local region start end out_dir regions_csv

    for entry in "${triplets[@]}"; do
        IFS='|' read -r sample subsample t0 t1 t2 <<< "$entry"
        timestamp="$t1"
        for chromosome in "${chroms[@]}"; do
            sample_tag="${RESOLUTION}_${MODEL_PATCH}_${organism}_${sample}_${subsample}_${timestamp}"
            out_tag="${RESOLUTION}_${organism}_${sample}_${subsample}_${timestamp}"
            prefix="${sample_tag}_${chromosome}"
            y_matrix="${INPUT_DIR}/${prefix}_y.npy"

            if [[ ! -f "$y_matrix" ]]; then
                echo "[run_flamingo] SKIP missing y: $y_matrix" >&2
                continue
            fi

            n_bins="$(python3 -c "import numpy as np; print(np.load('${y_matrix}', mmap_mode='r').shape[0])")"
            build_regions "$n_bins"
            if (( ${#REGIONS_ARR[@]} == 0 )); then
                continue
            fi
            echo "[run_flamingo] ${organism} ${sample}/${subsample}/${timestamp} chr${chromosome} n_bins=${n_bins} regions=${#REGIONS_ARR[@]} (${REGIONS_ARR[*]})"

            for method in "${METHODS[@]}"; do
                matrix_file="${INPUT_DIR}/${prefix}_${method}.npy"
                if [[ ! -f "$matrix_file" ]]; then
                    echo "[run_flamingo] SKIP missing: $matrix_file" >&2
                    continue
                fi
                for region in "${REGIONS_ARR[@]}"; do
                    start="${region%%:*}"
                    end="${region##*:}"
                    out_dir="${OUTPUT_ROOT}/${out_tag}_${chromosome}_${method}/region_${start}_${end}"
                    mkdir -p "$out_dir"
                    if [[ -f "${out_dir}/flamingo_structure.pdb" && -f "${out_dir}/flamingo_coords.tsv" ]]; then
                        echo "[run_flamingo] SKIP exists: ${organism} ${method} ${start}:${end}"
                        continue
                    fi
                    echo "[run_flamingo] ${organism} ${method} bins ${start}:${end} ($((end - start)) bins)"
                    if ! python3 "$SCRIPT_DIR/run_flamingo.py" \
                        --input "$matrix_file" \
                        --output_dir "$out_dir" \
                        --bin_size "$RESOLUTION" \
                        --domain_res "$DOMAIN_RES" \
                        --chrom "$chromosome" \
                        --start "$start" \
                        --end "$end" \
                        --n_thread "$N_THREAD" \
                        --rscript "$RSCRIPT" \
                        --force_large \
                        --norm_ref "$y_matrix"
                    then
                        echo "[run_flamingo] FAILED: ${organism} ${method} ${start}:${end}" >&2
                        rm -f "${out_dir}/flamingo_coords.tsv" "${out_dir}/flamingo_structure.pdb" "${out_dir}/flamingo_structure.vtk"
                        continue
                    fi
                done
            done

            regions_csv="$(IFS=,; echo "${REGIONS_ARR[*]}")"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$sample" "$subsample" "$timestamp" "$chromosome" "$out_tag" "$regions_csv" \
                >> "$JOBS_FILE"
        done
    done
}

JOBS_FILE="$(mktemp)"
trap 'rm -f "$JOBS_FILE"' EXIT

echo "[run_flamingo] human chroms=${HUMAN_CHROMS[*]}"
process_triplet_set human "${HUMAN_CHROMS[@]}"

echo "[run_flamingo] mouse chroms=${MOUSE_CHROMS[*]}"
process_triplet_set mouse "${MOUSE_CHROMS[@]}"

echo "[run_flamingo] structures done; computing SCC -> $SCC_CSV"
python3 "$SCRIPT_DIR/calculate_flamingo_scc.py" \
    --flamingo_root "$OUTPUT_ROOT" \
    --jobs_file "$JOBS_FILE" \
    --output_csv "$SCC_CSV"

echo "[run_flamingo] all done"
