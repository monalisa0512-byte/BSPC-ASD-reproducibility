#!/usr/bin/env python3
"""Fail-fast contract audit for the requested strict nested-window batch."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    paths = list(root.glob("*/*/seed_*/task_contract.json"))
    contracts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = {
        (model, "Full_Preprocessing", seed)
        for model in ("AttentionNet", "CNNLSTM", "PureLSTM")
        for seed in range(42, 47)
    } | {("AttentionNet", "No_Filtering", seed) for seed in range(42, 47)}
    observed = {(item["model"], item["config"], item["global_seed"]) for item in contracts}
    assert len(contracts) == 20, f"expected 20 contracts, found {len(contracts)}"
    assert observed == expected, f"task mismatch: {sorted(observed ^ expected)}"
    assert all(item["candidates_ms"] == [[2000, 1000], [1500, 750], [1000, 500], [2000, 500]] for item in contracts)
    assert all(item["epochs"] == 50 for item in contracts)
    assert all(item["outer_fold_seed_scheme"] == "global_seed_times_100000_plus_one_based_fold_id_v1" for item in contracts)
    assert all(item["candidate_seed_scheme"] == "fold_seed_times_10_plus_one_based_candidate_index_v1" for item in contracts)
    assert all(not item["window_selection_uses_outer_test"] for item in contracts)
    assert all(not item["standardization_uses_outer_test"] for item in contracts)
    assert all(item["same_condition_train_validation_test"] for item in contracts)
    print("PRELAUNCH_AUDIT_PASS tasks=20 candidates=4 epochs=50 seeds=42-46")
    print("tasks:", dict(sorted(Counter((item["model"], item["config"]) for item in contracts).items())))
    print("model settings:", sorted({(item["model"], item["batch_size"], item["learning_rate"]) for item in contracts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
