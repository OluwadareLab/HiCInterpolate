#!/usr/bin/env bash
# FLAMINGO 3D structures (same chroms/triplets as run_flamingo.sh), repeated N_RUNS times.
# Each run writes under OUTPUT_ROOT/run_XX/; SCC for all runs -> one CSV (with run column).
# Usage:
#   bash run_flamingo_20.sh
#   N_RUNS=20 bash run_flamingo_20.sh
#   CHROMS="10 11" bash run_flamingo_20.sh
#   MIN_BINS=2500 MAX_BINS=4000 bash run_flamingo_20.sh
set -euo pipefail

SCRIPT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate/analysis"
RSCRIPT="${RSCRIPT:-/home/hc0783.unt.ad.unt.edu/.conda/envs/flamingo/bin/Rscript}"
INPUT_DIR="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/config_25k_64"
OUTPUT_ROOT="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/flamingo_user_20"

RESOLUTION="${RESOLUTION:-25000}"
MODEL_PATCH="${MODEL_PATCH:-64}"
DOMAIN_RES="${DOMAIN_RES:-1000000}"
MIN_BINS="${MIN_BINS:-2500}"
MAX_BINS="${MAX_BINS:-4000}"
N_THREAD="${N_THREAD:-1}"
N_RUNS="${N_RUNS:-20}"
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
    echo "[run_flamingo_20] need 0 <= MIN_BINS < MAX_BINS (got ${MIN_BINS}, ${MAX_BINS})" >&2
    exit 1
fi
if (( N_RUNS < 1 )); then
    echo "[run_flamingo_20] N_RUNS must be >= 1 (got ${N_RUNS})" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "[run_flamingo_20] n_runs=${N_RUNS} bins [${MIN_BINS}, ${MAX_BINS}] @ ${RESOLUTION}"
echo "[run_flamingo_20] methods=${METHODS[*]}"
echo "[run_flamingo_20] human chroms=${HUMAN_CHROMS[*]}"
echo "[run_flamingo_20] mouse chroms=${MOUSE_CHROMS[*]}"
echo "[run_flamingo_20] scc_csv=${SCC_CSV}"

build_regions() {
    local n_bins="$1"
    REGIONS_ARR=()
    if (( n_bins <= MIN_BINS )); then
        echo "[run_flamingo_20] SKIP region: n_bins=${n_bins} <= MIN_BINS=${MIN_BINS}" >&2
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
    local run_root="$2"
    local jobs_file="$3"
    shift 3
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
                echo "[run_flamingo_20] SKIP missing y: $y_matrix" >&2
                continue
            fi

            n_bins="$(python3 -c "import numpy as np; print(np.load('${y_matrix}', mmap_mode='r').shape[0])")"
            build_regions "$n_bins"
            if (( ${#REGIONS_ARR[@]} == 0 )); then
                continue
            fi
            echo "[run_flamingo_20] ${organism} ${sample}/${subsample}/${timestamp} chr${chromosome} n_bins=${n_bins} regions=${#REGIONS_ARR[@]} (${REGIONS_ARR[*]})"

            for method in "${METHODS[@]}"; do
                matrix_file="${INPUT_DIR}/${prefix}_${method}.npy"
                if [[ ! -f "$matrix_file" ]]; then
                    echo "[run_flamingo_20] SKIP missing: $matrix_file" >&2
                    continue
                fi
                for region in "${REGIONS_ARR[@]}"; do
                    start="${region%%:*}"
                    end="${region##*:}"
                    out_dir="${run_root}/${out_tag}_${chromosome}_${method}/region_${start}_${end}"
                    mkdir -p "$out_dir"
                    if [[ -f "${out_dir}/flamingo_structure.pdb" && -f "${out_dir}/flamingo_coords.tsv" ]]; then
                        echo "[run_flamingo_20] SKIP exists: ${organism} ${method} ${start}:${end}"
                        continue
                    fi
                    echo "[run_flamingo_20] ${organism} ${method} bins ${start}:${end} ($((end - start)) bins)"
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
                        echo "[run_flamingo_20] FAILED: ${organism} ${method} ${start}:${end}" >&2
                        rm -f "${out_dir}/flamingo_coords.tsv" "${out_dir}/flamingo_structure.pdb" "${out_dir}/flamingo_structure.vtk"
                        continue
                    fi
                done
            done

            regions_csv="$(IFS=,; echo "${REGIONS_ARR[*]}")"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$sample" "$subsample" "$timestamp" "$chromosome" "$out_tag" "$regions_csv" \
                >> "$jobs_file"
        done
    done
}

SCC_PARTS_DIR="$(mktemp -d)"
trap 'rm -rf "$SCC_PARTS_DIR"' EXIT

for (( run_i = 1; run_i <= N_RUNS; run_i++ )); do
    run_tag="$(printf 'run_%02d' "$run_i")"
    run_root="${OUTPUT_ROOT}/${run_tag}"
    mkdir -p "$run_root"
    JOBS_FILE="$(mktemp)"
    echo "[run_flamingo_20] ===== ${run_tag} / ${N_RUNS} -> ${run_root} ====="

    process_triplet_set human "$run_root" "$JOBS_FILE" "${HUMAN_CHROMS[@]}"
    process_triplet_set mouse "$run_root" "$JOBS_FILE" "${MOUSE_CHROMS[@]}"

    part_csv="${SCC_PARTS_DIR}/${run_tag}.csv"
    if [[ -s "$JOBS_FILE" ]]; then
        echo "[run_flamingo_20] ${run_tag} structures done; SCC part -> $part_csv"
        python3 "$SCRIPT_DIR/calculate_flamingo_scc.py" \
            --flamingo_root "$run_root" \
            --jobs_file "$JOBS_FILE" \
            --output_csv "$part_csv"
    else
        echo "[run_flamingo_20] ${run_tag} no jobs; skip SCC" >&2
    fi
    rm -f "$JOBS_FILE"
done

echo "[run_flamingo_20] merging SCC parts -> $SCC_CSV"
python3 - "$SCC_PARTS_DIR" "$SCC_CSV" <<'PY'
import csv
import glob
import os
import sys

parts_dir, out_csv = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(os.path.join(parts_dir, "run_*.csv")))
if not paths:
    print(f"[run_flamingo_20] no SCC parts under {parts_dir}", flush=True)
    sys.exit(1)

fieldnames = None
rows = []
for path in paths:
    run = os.path.splitext(os.path.basename(path))[0]
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = ["run"] + list(reader.fieldnames)
        for row in reader:
            rows.append({"run": run, **row})

os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[run_flamingo_20] wrote {out_csv} ({len(rows)} rows from {len(paths)} runs)", flush=True)
PY

echo "[run_flamingo_20] all ${N_RUNS} runs done"
