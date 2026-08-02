#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-$REPOSITORY/data/eyesdata_processed_57}"
BASE="${BASE:-$REPOSITORY/run_outputs/fixed_multiseed_strict}"
OUTPUT_ROOT="$BASE/results"
LOG_ROOT="$BASE/logs"
GATE_DIR="$BASE/reproducibility_gate"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
EPOCHS="${EPOCHS:-50}"

MODELS=(AttentionNet CNNLSTM PureLSTM)
CONFIGS=(Full_Preprocessing Linear_Interpolation Without_Blink_Expansion No_Filtering Without_Mask_Features)
SEEDS=(42 43 44 45 46)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$EXPERIMENT_DIR"

if [[ ! -f "$GATE_DIR/PASS" ]]; then
  echo "Refusing formal launch: reproducibility gate PASS is missing at $GATE_DIR/PASS" >&2
  exit 20
fi
if ! (cd "$EXPERIMENT_DIR" && sha256sum -c "$GATE_DIR/source.sha256"); then
  echo "Refusing formal launch: source files changed after reproducibility gate" >&2
  exit 21
fi

run_job() {
  local model="$1"
  local config="$2"
  local seed="$3"
  local tag="${model}__${config}__seed_${seed}"
  local log_file="$LOG_ROOT/${tag}.log"
  local exit_file="$LOG_ROOT/${tag}.exit"
  rm -f "$exit_file"
  echo "[$(date '+%F %T')] START $tag" >>"$LOG_ROOT/launcher.log"
  (
    export PYTHONHASHSEED="$seed"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    "$PYTHON_BIN" -u run_fixed_multiseed_fold_strict.py \
      --model "$model" \
      --config "$config" \
      --global-seed "$seed" \
      --data-folder "$DATA_DIR" \
      --output-root "$OUTPUT_ROOT" \
      --epochs "$EPOCHS" \
      >"$log_file" 2>&1
    status=$?
    echo "$status" >"$exit_file"
    echo "[$(date '+%F %T')] END $tag exit=$status" >>"$LOG_ROOT/launcher.log"
    exit "$status"
  ) &
}

echo "[$(date '+%F %T')] FORMAL_BATCH_BEGIN max_parallel=$MAX_PARALLEL" >>"$LOG_ROOT/launcher.log"
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
        wait -n || true
      done
      run_job "$model" "$config" "$seed"
    done
  done
done

wait || true

failed=0
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      tag="${model}__${config}__seed_${seed}"
      exit_file="$LOG_ROOT/${tag}.exit"
      complete_file="$OUTPUT_ROOT/$model/$config/seed_${seed}/COMPLETE"
      if [[ ! -f "$complete_file" ]] || [[ ! -f "$exit_file" ]] || [[ "$(<"$exit_file")" != "0" ]]; then
        failed=$((failed + 1))
      fi
    done
  done
done

"$PYTHON_BIN" report_fixed_multiseed_progress.py \
  --output-root "$OUTPUT_ROOT" \
  --write-snapshot \
  >"$LOG_ROOT/final_progress.log" 2>&1 || true

if [[ "$failed" == "0" ]]; then
  "$PYTHON_BIN" aggregate_fixed_multiseed_strict.py \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$BASE/aggregate" \
    >"$LOG_ROOT/aggregate.log" 2>&1 || failed=1
fi

echo "[$(date '+%F %T')] FORMAL_BATCH_END failed=$failed" | tee -a "$LOG_ROOT/launcher.log"
exit "$failed"
