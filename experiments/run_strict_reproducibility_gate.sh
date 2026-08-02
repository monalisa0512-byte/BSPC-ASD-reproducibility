#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-$REPOSITORY/data/eyesdata_processed_57}"
BASE="${BASE:-$REPOSITORY/run_outputs/fixed_multiseed_strict}"
FORMAL_ROOT="$BASE/results"
REPEAT_ROOT="$BASE/reproducibility_gate/repeat_results"
GATE_DIR="$BASE/reproducibility_gate"
LOG_DIR="$GATE_DIR/logs"

mkdir -p "$FORMAL_ROOT" "$REPEAT_ROOT" "$LOG_DIR"
cd "$EXPERIMENT_DIR"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

COMMON_ARGS=(
  --model AttentionNet
  --config Full_Preprocessing
  --global-seed 42
  --data-folder "$DATA_DIR"
  --epochs 50
)

echo "[$(date '+%F %T')] Gate run A: official task" | tee "$LOG_DIR/gate.log"
"$PYTHON_BIN" -u run_fixed_multiseed_fold_strict.py \
  "${COMMON_ARGS[@]}" \
  --output-root "$FORMAL_ROOT" \
  >"$LOG_DIR/run_a.log" 2>&1

echo "[$(date '+%F %T')] Gate run B: independent repeat" | tee -a "$LOG_DIR/gate.log"
"$PYTHON_BIN" -u run_fixed_multiseed_fold_strict.py \
  "${COMMON_ARGS[@]}" \
  --output-root "$REPEAT_ROOT" \
  >"$LOG_DIR/run_b.log" 2>&1

RUN_A="$FORMAL_ROOT/AttentionNet/Full_Preprocessing/seed_42"
RUN_B="$REPEAT_ROOT/AttentionNet/Full_Preprocessing/seed_42"
"$PYTHON_BIN" verify_strict_task_reproduction.py \
  --run-a "$RUN_A" \
  --run-b "$RUN_B" \
  --output-dir "$GATE_DIR" \
  | tee -a "$LOG_DIR/gate.log"

sha256sum \
  run_fixed_multiseed_fold_strict.py \
  run_attentionnet_preprocessing_ablation_pipeline_fixed.py \
  run_cnnlstm_preprocessing_ablation_pipeline_fixed.py \
  run_purelstm_preprocessing_ablation_pipeline_fixed.py \
  >"$GATE_DIR/source.sha256"

echo "[$(date '+%F %T')] REPRODUCIBILITY_GATE_PASS" | tee -a "$LOG_DIR/gate.log"
