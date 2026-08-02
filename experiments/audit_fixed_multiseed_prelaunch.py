#!/usr/bin/env python3
"""Fail-closed prelaunch audit for the strict 75-task fixed-window study."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path

import pandas as pd

import run_fixed_multiseed_fold_strict as strict


EXPECTED_CONFIGS = {
    "Full_Preprocessing": {
        "pre_ms": 80,
        "post_ms": 160,
        "impute_method": "pchip",
        "use_mask_features": True,
        "trial_intercept": True,
        "window_intercept": True,
        "max_trial_missing_rate": 0.8,
        "max_missing_rate": 0.6,
        "max_continuous_frames": 60,
    },
    "Linear_Interpolation": {
        "pre_ms": 80,
        "post_ms": 160,
        "impute_method": "linear",
        "use_mask_features": True,
        "trial_intercept": True,
        "window_intercept": True,
        "max_trial_missing_rate": 0.8,
        "max_missing_rate": 0.6,
        "max_continuous_frames": 60,
    },
    "Without_Blink_Expansion": {
        "pre_ms": 0,
        "post_ms": 0,
        "impute_method": "pchip",
        "use_mask_features": True,
        "trial_intercept": True,
        "window_intercept": True,
        "max_trial_missing_rate": 0.8,
        "max_missing_rate": 0.6,
        "max_continuous_frames": 60,
    },
    "No_Filtering": {
        "pre_ms": 80,
        "post_ms": 160,
        "impute_method": "pchip",
        "use_mask_features": True,
        "trial_intercept": False,
        "window_intercept": False,
        "max_trial_missing_rate": 1.0,
        "max_missing_rate": 1.0,
        "max_continuous_frames": "inf",
    },
    "Without_Mask_Features": {
        "pre_ms": 80,
        "post_ms": 160,
        "impute_method": "pchip",
        "use_mask_features": False,
        "trial_intercept": True,
        "window_intercept": True,
        "max_trial_missing_rate": 0.8,
        "max_missing_rate": 0.6,
        "max_continuous_frames": 60,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def normalized_config(config) -> dict:
    continuous = config.max_continuous_frames
    if continuous == float("inf"):
        continuous = "inf"
    return {
        "pre_ms": config.pre_ms,
        "post_ms": config.post_ms,
        "impute_method": config.impute_method,
        "use_mask_features": config.use_mask_features,
        "trial_intercept": config.trial_intercept,
        "window_intercept": config.window_intercept,
        "max_trial_missing_rate": config.max_trial_missing_rate,
        "max_missing_rate": config.max_missing_rate,
        "max_continuous_frames": continuous,
    }


def main() -> None:
    args = parse_args()
    formal_root = Path(args.formal_root).resolve()
    gate_dir = Path(args.gate_dir).resolve()
    output = Path(args.output).resolve()
    errors = []

    matrix = [
        (model, config, seed)
        for seed in range(42, 47)
        for model in strict.MODELS
        for config in strict.CONFIGS
    ]
    if len(matrix) != 75 or len(set(matrix)) != 75:
        errors.append(f"Task matrix is not 75 unique triples: {len(matrix)} / {len(set(matrix))}")

    os.environ["BSPC_FIXED_RESULTS_FOLDER"] = str(output.parent / "audit_import_logs")
    module_configs = {}
    modules = {}
    for model, (module_name, _) in strict.MODELS.items():
        module = importlib.import_module(module_name)
        modules[model] = module
        configs = {config.name: normalized_config(config) for config in module.ABLATION_CONFIGS}
        module_configs[model] = configs
        if configs != EXPECTED_CONFIGS:
            errors.append(f"Configuration mismatch for {model}: {configs}")
        source = inspect.getsource(module.extract_single_pid_data)
        if "config.max_missing_rate" not in source or "config.max_continuous_frames" not in source:
            errors.append(f"Validation extraction is not condition-aware for {model}")

    runner_source = inspect.getsource(strict.run_fold)
    required_runner_tokens = (
        "seed_everything(current_seed)",
        "random_state=current_seed",
        "extract_single_pid_data(full_df, val_pid, scaler, config)",
        "create_dataset_from_full_data(",
        "sampler_generator.manual_seed(current_seed)",
        "scaler_amp = torch.cuda.amp.GradScaler()",
    )
    for token in required_runner_tokens:
        if token not in runner_source:
            errors.append(f"Strict runner token missing: {token}")

    if not (gate_dir / "PASS").exists():
        errors.append("Reproducibility gate PASS is missing")
    verification_path = gate_dir / "reproducibility_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or verification.get("errors"):
        errors.append(f"Reproducibility verification did not pass cleanly: {verification}")

    official = formal_root / "AttentionNet" / "Full_Preprocessing" / "seed_42"
    predictions = pd.read_csv(official / "fold_predictions.csv", encoding="utf-8-sig")
    if len(predictions) != 57 or predictions["test_pid"].nunique() != 57:
        errors.append("Official gate task does not contain 57 unique LOSO folds")
    predictions = predictions.sort_values("fold_id").reset_index(drop=True)
    pid_to_label = dict(
        zip(predictions["test_pid"].astype(int), predictions["true_label"].astype(int))
    )
    all_pids = sorted(pid_to_label)

    checked_splits = 0
    cross_model_split_mismatches = 0
    for seed in range(42, 47):
        for fold_index, test_pid in enumerate(all_pids, start=1):
            expected_seed = strict.fold_seed(seed, fold_index)
            if expected_seed != seed * 100_000 + fold_index:
                errors.append(f"Fold seed mismatch for seed={seed}, fold={fold_index}")
            reference = None
            for model, module in modules.items():
                pool = [pid for pid in all_pids if pid != test_pid]
                labels = [pid_to_label[pid] for pid in pool]
                strict.seed_everything(expected_seed)
                train_pids, val_pids = module.get_stratified_train_val_pids(
                    pool, labels, test_size=0.15, random_state=expected_seed
                )
                split = (tuple(train_pids), tuple(val_pids), int(test_pid), expected_seed)
                if set(train_pids) & set(val_pids):
                    errors.append(f"Train/validation overlap: {model}, seed={seed}, fold={fold_index}")
                if test_pid in train_pids or test_pid in val_pids:
                    errors.append(f"Test participant leakage: {model}, seed={seed}, fold={fold_index}")
                if len(train_pids) != 47 or len(val_pids) != 9:
                    errors.append(
                        f"Unexpected split size: {model}, seed={seed}, fold={fold_index}, "
                        f"train={len(train_pids)}, val={len(val_pids)}"
                    )
                if reference is None:
                    reference = split
                elif split != reference:
                    cross_model_split_mismatches += 1
                checked_splits += 1

    for fold_path in sorted((official / "folds").glob("fold_*.json")):
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        fold_id = int(fold["fold_id"])
        if int(fold["fold_seed"]) != strict.fold_seed(42, fold_id):
            errors.append(f"Saved gate fold seed mismatch: {fold_path.name}")
        pool = [pid for pid in all_pids if pid != int(fold["test_pid"])]
        labels = [pid_to_label[pid] for pid in pool]
        train_pids, val_pids = modules["AttentionNet"].get_stratified_train_val_pids(
            pool,
            labels,
            test_size=0.15,
            random_state=int(fold["fold_seed"]),
        )
        if fold["train_pids"] != train_pids or fold["val_pids"] != val_pids:
            errors.append(f"Saved split differs from recomputed split: {fold_path.name}")

    if cross_model_split_mismatches:
        errors.append(f"Cross-model split mismatches: {cross_model_split_mismatches}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "task_matrix_count": len(matrix),
        "task_matrix_unique_count": len(set(matrix)),
        "models": list(strict.MODELS),
        "conditions": list(strict.CONFIGS),
        "global_seeds": list(range(42, 47)),
        "expected_total_folds": 75 * 57,
        "fold_seed_scheme": strict.FOLD_SEED_SCHEME,
        "checked_model_seed_fold_splits": checked_splits,
        "cross_model_split_mismatches": cross_model_split_mismatches,
        "same_split_applies_to_all_five_conditions": True,
        "module_configs": module_configs,
        "gate_verification": verification,
        "official_gate_accuracy": float((predictions["true_label"] == predictions["pred_label"]).mean()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise SystemExit("Prelaunch audit failed:\n- " + "\n- ".join(errors))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
