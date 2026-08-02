#!/usr/bin/env bash
set -euo pipefail

# Launch the requested nested-window five-seed tasks as independent, resumable
# processes.  One task = one (model, preprocessing condition, global seed).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_FOLDER="${DATA_FOLDER:-${REPOSITORY}/data/eyesdata_processed_57}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY}/run_outputs/nested_multiseed_strict}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
EPOCHS="${EPOCHS:-50}"
PYTHON="${PYTHON:-python}"
SEEDS=(42 43 44 45 46)

mkdir -p "${OUTPUT_ROOT}/_launcher_logs" "${OUTPUT_ROOT}/_launcher_state"

TASKS=()
for model in AttentionNet CNNLSTM PureLSTM; do
  for seed in "${SEEDS[@]}"; do
    TASKS+=("${model}|Full_Preprocessing|${seed}")
  done
done
for seed in "${SEEDS[@]}"; do
  TASKS+=("AttentionNet|No_Filtering|${seed}")
done

task_dir() {
  local model="$1" config="$2" seed="$3"
  printf '%s/%s/%s/seed_%s' "${OUTPUT_ROOT}" "${model}" "${config}" "${seed}"
}

is_alive() {
  local pid_file="$1"
  [[ -s "${pid_file}" ]] || return 1
  local pid
  pid="$(<"${pid_file}")"
  kill -0 "${pid}" 2>/dev/null
}

active_count() {
  local count=0 task dir
  for task in "${TASKS[@]}"; do
    IFS='|' read -r model config seed <<<"${task}"
    dir="$(task_dir "${model}" "${config}" "${seed}")"
    if is_alive "${dir}/task.pid"; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

wait_for_slot() {
  while (( $(active_count) >= MAX_PARALLEL )); do
    sleep 15
  done
}

start_task() {
  local model="$1" config="$2" seed="$3" dir="$4"
  mkdir -p "${dir}/folds"
  if [[ -f "${dir}/COMPLETE" ]]; then
    return 0
  fi
  if is_alive "${dir}/task.pid"; then
    return 0
  fi
  rm -f "${dir}/failure.json"
  local log="${OUTPUT_ROOT}/_launcher_logs/${model}__${config}__seed_${seed}.log"
  echo "[$(date '+%F %T %z')] starting ${model} ${config} seed=${seed}" >>"${OUTPUT_ROOT}/_launcher_state/launcher.log"
  nohup setsid "${PYTHON}" "${SCRIPT_DIR}/run_nested_multiseed_fold_strict.py" \
    --model "${model}" --config "${config}" --global-seed "${seed}" \
    --data-folder "${DATA_FOLDER}" --output-root "${OUTPUT_ROOT}" --epochs "${EPOCHS}" \
    >"${log}" 2>&1 < /dev/null &
  echo $! >"${dir}/task.pid"
}

echo "Nested strict launcher"
echo "output=${OUTPUT_ROOT}"
echo "data=${DATA_FOLDER}"
echo "max_parallel=${MAX_PARALLEL} epochs=${EPOCHS} tasks=${#TASKS[@]}"

for task in "${TASKS[@]}"; do
  IFS='|' read -r model config seed <<<"${task}"
  dir="$(task_dir "${model}" "${config}" "${seed}")"
  wait_for_slot
  start_task "${model}" "${config}" "${seed}" "${dir}"
done

echo "[$(date '+%F %T %z')] all ${#TASKS[@]} tasks submitted" >>"${OUTPUT_ROOT}/_launcher_state/launcher.log"
echo "All tasks submitted. Use report_nested_multiseed_progress.py to monitor progress."
