#!/usr/bin/env bash
set -euo pipefail

# Portable launcher for the reviewer-requested random-seed sensitivity run.
ROOT="${BSPC_PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${BSPC_DATA_DIR:-$ROOT/data/eyesdata_processed_57}"
OUT="$ROOT/results/seed_sensitivity_fixed_1000_500_server_v2"
PYTHON="${PYTHON:-python}"
SEEDS=(42 43 44 45 46)

cd "$ROOT"
mkdir -p "$OUT"

# Improve deterministic CUDA behaviour where supported. The experiment script
# itself also fixes Python/NumPy/PyTorch/CUDA seeds and deterministic cuDNN.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in "${SEEDS[@]}"; do
    echo "===== fixed-window AttentionNet seed ${seed} ====="
    PYTHONHASHSEED="$seed" "$PYTHON" experiments/run_fixed_window_seed.py \
        --seed "$seed" \
        --data-folder "$DATA" \
        --output-root "$OUT" \
        --window-size 1000 \
        --stride 500 \
        --models AttentionNet \
        2>&1 | tee "$OUT/seed_${seed}.log"
done

"$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import pandas as pd
import sys

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob('seed_*/summary.csv')):
    seed = int(path.parent.name.split('_')[1])
    df = pd.read_csv(path)
    df.insert(0, 'seed', seed)
    rows.append(df)

if not rows:
    raise SystemExit('No seed summaries found; the experiment did not finish.')

summary = pd.concat(rows, ignore_index=True)
summary.to_csv(root / 'seed_summary_all.csv', index=False)
metrics = summary[summary['model'].eq('AttentionNet')]
stats = metrics[['accuracy', 'f1']].agg(['mean', 'std', 'min', 'max'])
stats.to_csv(root / 'seed_summary_stats.csv')

print('\n===== aggregate seed summary =====')
print(metrics.to_string(index=False))
print('\n===== mean / SD / range =====')
print(stats)
print(f'\nSaved: {root / "seed_summary_all.csv"}')
print(f'Saved: {root / "seed_summary_stats.csv"}')
PY
