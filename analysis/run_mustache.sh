#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/config_25k_64"
OUTPUT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/mustache"
SCRIPT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate/analysis"

RESOLUTIONS=(25000)
PATCHES=(64)

HUMAN_CHROMS=(10 11 15 16 20 21)
MOUSE_CHROMS=(10 15 19)

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

get_res_tag() {
    case "$1" in
        25000) printf '25k' ;;
        10000) printf '10k' ;;
        5000) printf '5k' ;;
        *) echo "Unsupported resolution: $1" >&2; return 1 ;;
    esac
}

mkdir -p "$OUTPUT_DIR"
echo "[run_mustache] output root: $OUTPUT_DIR"

for resolution in "${RESOLUTIONS[@]}"; do
    res_tag="$(get_res_tag "$resolution")"
    echo "[run_mustache] resolution: $resolution ($res_tag)"
    for model_patch in "${PATCHES[@]}"; do
        echo "[run_mustache] model patch: $model_patch"
        for entry in "${HUMAN_TRIPLETS[@]}"; do
            IFS='|' read -r sample subsample t0 t1 t2 <<< "$entry"
            for chromosome in "${HUMAN_CHROMS[@]}"; do
                for algo in y pred of linear 4dmax; do
                    matrix_file="${INPUT_DIR}/${resolution}_${model_patch}_human_${sample}_${subsample}_${t1}_${chromosome}_${algo}.npy"
                    sub_out_dir="${OUTPUT_DIR}/${resolution}_human_${sample}_${subsample}_${t1}_${chromosome}_${algo}"
                    mkdir -p "$sub_out_dir"
                    echo "[run_mustache] human ${sample}/${subsample} chr${chromosome}: $matrix_file"

                    if [[ ! -f "$matrix_file" ]]; then
                        echo "[run_mustache] SKIP missing: $matrix_file" >&2
                        continue
                    fi

                    python3 "$SCRIPT_DIR/run_mustache.py" \
                        --matrix_file "$matrix_file" \
                        --output_dir "$sub_out_dir" \
                        --bin_size "$resolution" \
                        --chrom "$chromosome" \
                        --keep_cool
                done
            done
        done

        for entry in "${MOUSE_TRIPLETS[@]}"; do
            IFS='|' read -r sample subsample t0 t1 t2 <<< "$entry"
            for chromosome in "${MOUSE_CHROMS[@]}"; do
                for algo in y pred of linear 4dmax; do
                    matrix_file="${INPUT_DIR}/${resolution}_${model_patch}_mouse_${sample}_${subsample}_${t1}_${chromosome}_${algo}.npy"
                    sub_out_dir="${OUTPUT_DIR}/${resolution}_mouse_${sample}_${subsample}_${t1}_${chromosome}_${algo}"
                    mkdir -p "$sub_out_dir"
                    echo "[run_mustache] mouse ${sample}/${subsample} chr${chromosome}: $matrix_file"

                    if [[ ! -f "$matrix_file" ]]; then
                        echo "[run_mustache] SKIP missing: $matrix_file" >&2
                        continue
                    fi

                    python3 "$SCRIPT_DIR/run_mustache.py" \
                        --matrix_file "$matrix_file" \
                        --output_dir "$sub_out_dir" \
                        --bin_size "$resolution" \
                        --chrom "$chromosome" \
                        --keep_cool
                done
            done
        done
    done
done

echo "[run_mustache] done"
