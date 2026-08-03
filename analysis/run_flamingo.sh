#!/usr/bin/env bash
# FLAMINGO on specified bin regions for all methods.
# Usage:
#   bash run_flamingo.sh
#   REGIONS="0:1338,1338:2676" bash run_flamingo.sh
#   bash run_flamingo.sh --region 120:240 --region 500:700
set -euo pipefail

SCRIPT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate/analysis"
RSCRIPT="${RSCRIPT:-/home/hc0783.unt.ad.unt.edu/.conda/envs/flamingo/bin/Rscript}"
INPUT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/config_25k_64"
OUTPUT_ROOT="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/flamingo_user"

BIN_SIZE=25000
DOMAIN_RES=1000000
CHROM="${CHROM:-10}"
N_THREAD="${N_THREAD:-1}"
METHODS=(${METHODS:-y pred of linear 4dmax})
SAMPLE_TAG="${SAMPLE_TAG:-25000_64_human_dmso_control_dmso_control_60m}"
OUT_TAG="${OUT_TAG:-25000_human_dmso_control_dmso_control_60m}"

# Default regions (override with REGIONS=start:end,... or --region start:end)
DEFAULT_REGIONS=("0:1338" "1338:2676" "2676:4014" "4014:5352")
REGIONS_ARR=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --region|-r)
            REGIONS_ARR+=("$2")
            shift 2
            ;;
        --chrom)
            CHROM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--region START:END]... [--chrom N]"
            echo "  or:  REGIONS='START:END,START:END' $0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${#REGIONS_ARR[@]} -eq 0 ]]; then
    if [[ -n "${REGIONS:-}" ]]; then
        IFS=',' read -r -a REGIONS_ARR <<< "$REGIONS"
    else
        REGIONS_ARR=("${DEFAULT_REGIONS[@]}")
    fi
fi

PREFIX="${SAMPLE_TAG}_${CHROM}"
Y_MATRIX="${INPUT_DIR}/${PREFIX}_y.npy"
echo "[run_flamingo] chr${CHROM} methods=${METHODS[*]} regions=${REGIONS_ARR[*]}"
echo "[run_flamingo] norm_ref (y): $Y_MATRIX"

for method in "${METHODS[@]}"; do
    matrix_file="${INPUT_DIR}/${PREFIX}_${method}.npy"
    if [[ ! -f "$matrix_file" ]]; then
        echo "[run_flamingo] SKIP missing: $matrix_file" >&2
        continue
    fi
    for region in "${REGIONS_ARR[@]}"; do
        region="${region// /}"
        start="${region%%:*}"
        end="${region##*:}"
        if [[ -z "$start" || -z "$end" || "$start" == "$end" ]]; then
            echo "[run_flamingo] bad region: $region (want START:END)" >&2
            exit 1
        fi
        out_dir="${OUTPUT_ROOT}/${OUT_TAG}_${CHROM}_${method}/region_${start}_${end}"
        mkdir -p "$out_dir"
        if [[ -f "${out_dir}/flamingo_structure.pdb" && -f "${out_dir}/flamingo_coords.tsv" ]]; then
            echo "[run_flamingo] SKIP exists: ${method} ${start}:${end}"
            continue
        fi
        echo "[run_flamingo] ${method} bins ${start}:${end}"
        if ! python3 "$SCRIPT_DIR/run_flamingo.py" \
            --input "$matrix_file" \
            --output_dir "$out_dir" \
            --bin_size "$BIN_SIZE" \
            --domain_res "$DOMAIN_RES" \
            --chrom "$CHROM" \
            --start "$start" \
            --end "$end" \
            --n_thread "$N_THREAD" \
            --rscript "$RSCRIPT" \
            --norm_ref "$Y_MATRIX"
        then
            echo "[run_flamingo] FAILED: ${method} ${start}:${end}" >&2
            rm -f "${out_dir}/flamingo_coords.tsv" "${out_dir}/flamingo_structure.pdb" "${out_dir}/flamingo_structure.vtk"
            continue
        fi
    done
done

echo "[run_flamingo] all regions done"
