#!/usr/bin/env python3
"""Aggregate the 20 strict nested-window tasks without seed selection."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


MODELS = ("AttentionNet", "CNNLSTM", "PureLSTM")
SEEDS = (42, 43, 44, 45, 46)
FULL = "Full_Preprocessing"
NO_FILTER = "No_Filtering"
CANDIDATES = ((2000, 1000), (1500, 750), (1000, 500), (2000, 500))
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 20260801


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_predictions(root: Path, model: str, config: str, seed: int) -> pd.DataFrame:
    task = root / model / config / f"seed_{seed}"
    if not (task / "COMPLETE").exists():
        raise FileNotFoundError(f"Incomplete task: {task}")
    frame = pd.read_csv(task / "fold_predictions.csv", encoding="utf-8-sig")
    frame = frame.sort_values("fold_id").reset_index(drop=True)
    if len(frame) != 57 or frame["test_pid"].nunique() != 57:
        raise ValueError(f"Expected 57 unique participants: {task}")
    return frame


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    lower = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return float(min(1.0, 2 * lower))


def f1_rows(y_true: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    tp = np.sum((y_true == 1) & (prediction == 1), axis=2)
    fp = np.sum((y_true == 0) & (prediction == 1), axis=2)
    fn = np.sum((y_true == 1) & (prediction == 0), axis=2)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)


def crossed_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    seed_offset: int,
    family_size: int = 1,
) -> dict[str, float]:
    """Crossed resampling of the 5 pipeline seeds and 57 participants."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    n_seeds, n_participants = pred_a.shape
    sampled_seeds = rng.integers(0, n_seeds, size=(BOOTSTRAP_ITERATIONS, n_seeds))
    sampled_participants = rng.integers(0, n_participants, size=(BOOTSTRAP_ITERATIONS, n_participants))
    pa = pred_a[sampled_seeds[:, :, None], sampled_participants[:, None, :]]
    pb = pred_b[sampled_seeds[:, :, None], sampled_participants[:, None, :]]
    yt = y_true[sampled_participants][:, None, :]
    accuracy_delta = np.mean(yt == pa, axis=(1, 2)) - np.mean(yt == pb, axis=(1, 2))
    f1_delta = np.mean(f1_rows(yt, pa), axis=1) - np.mean(f1_rows(yt, pb), axis=1)
    acc_low, acc_high = np.percentile(accuracy_delta, [2.5, 97.5])
    f1_low, f1_high = np.percentile(f1_delta, [2.5, 97.5])
    adjusted_tail = 100 * 0.05 / (2 * family_size)
    acc_family_low, acc_family_high = np.percentile(accuracy_delta, [adjusted_tail, 100 - adjusted_tail])
    f1_family_low, f1_family_high = np.percentile(f1_delta, [adjusted_tail, 100 - adjusted_tail])
    return {
        "accuracy_ci95_low": float(acc_low),
        "accuracy_ci95_high": float(acc_high),
        "f1_ci95_low": float(f1_low),
        "f1_ci95_high": float(f1_high),
        "accuracy_familywise_ci_low": float(acc_family_low),
        "accuracy_familywise_ci_high": float(acc_family_high),
        "f1_familywise_ci_low": float(f1_family_low),
        "f1_familywise_ci_high": float(f1_family_high),
        "family_size": family_size,
    }


def matrix(tasks: dict[tuple[str, str, int], pd.DataFrame], model: str, config: str) -> np.ndarray:
    return np.stack([tasks[(model, config, seed)]["pred_label"].to_numpy(dtype=int) for seed in SEEDS])


def paired_row(
    *,
    comparison: str,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    seed_offset: int,
    family_size: int,
) -> dict[str, object]:
    accuracy_delta = []
    f1_delta = []
    mcnemar = {}
    discordant = {}
    for index, seed in enumerate(SEEDS):
        a = pred_a[index]
        b = pred_b[index]
        correct_a = a == y_true
        correct_b = b == y_true
        a_only = int(np.sum(correct_a & ~correct_b))
        b_only = int(np.sum(~correct_a & correct_b))
        accuracy_delta.append(float(np.mean(correct_a) - np.mean(correct_b)))
        f1_delta.append(float(f1_score(y_true, a, zero_division=0) - f1_score(y_true, b, zero_division=0)))
        mcnemar[str(seed)] = exact_mcnemar_p(a_only, b_only)
        discordant[str(seed)] = {"a_only_correct": a_only, "b_only_correct": b_only}
    accuracy_array = np.asarray(accuracy_delta)
    f1_array = np.asarray(f1_delta)
    return {
        "comparison": comparison,
        "n_participants": len(y_true),
        "n_pipeline_seeds": len(SEEDS),
        "delta_accuracy_mean": float(np.mean(accuracy_array)),
        "delta_accuracy_sd_across_seeds": float(np.std(accuracy_array, ddof=1)),
        "delta_f1_mean": float(np.mean(f1_array)),
        "delta_f1_sd_across_seeds": float(np.std(f1_array, ddof=1)),
        "seeds_a_higher_accuracy": int(np.sum(accuracy_array > 0)),
        "seeds_equal_accuracy": int(np.sum(accuracy_array == 0)),
        "seeds_a_lower_accuracy": int(np.sum(accuracy_array < 0)),
        "seed_delta_accuracy": json.dumps(dict(zip(SEEDS, accuracy_delta))),
        "seed_delta_f1": json.dumps(dict(zip(SEEDS, f1_delta))),
        "per_seed_exact_mcnemar_p_diagnostic": json.dumps(mcnemar),
        "per_seed_discordant_pairs": json.dumps(discordant),
        **crossed_bootstrap(y_true, pred_a, pred_b, seed_offset, family_size),
    }


def fold_signature(path: Path) -> tuple:
    fold = json.loads(path.read_text(encoding="utf-8"))
    candidate_signature = tuple(
        (item["candidate_index"], item.get("candidate_seed"), item["window_size_ms"], item["stride_ms"])
        for item in fold["candidate_summaries"]
    )
    return (
        fold["fold_id"],
        fold["test_pid"],
        tuple(fold["train_pids"]),
        tuple(fold["val_pids"]),
        fold["fold_seed"],
        candidate_signature,
    )


def main() -> int:
    args = parse_args()
    root = Path(args.input_root).resolve()
    fixed_root = Path(args.fixed_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    requested = [(model, FULL) for model in MODELS] + [("AttentionNet", NO_FILTER)]
    tasks: dict[tuple[str, str, int], pd.DataFrame] = {}
    metric_rows = []
    reference_pid = None
    reference_truth = None
    split_reference: dict[int, list[tuple]] = {}
    window_rows = []
    all_candidates_usable = True

    for model, config in requested:
        for seed in SEEDS:
            frame = load_predictions(root, model, config, seed)
            pid = frame["test_pid"].astype(int).tolist()
            truth = frame["true_label"].astype(int).tolist()
            if reference_pid is None:
                reference_pid, reference_truth = pid, truth
            if pid != reference_pid or truth != reference_truth:
                raise ValueError(f"Participant alignment mismatch: {model}/{config}/seed_{seed}")
            tasks[(model, config, seed)] = frame
            metric_rows.append({
                "model": model,
                "config": config,
                "global_seed": seed,
                "accuracy": accuracy_score(frame["true_label"], frame["pred_label"]),
                "f1": f1_score(frame["true_label"], frame["pred_label"], zero_division=0),
            })
            fold_paths = sorted((root / model / config / f"seed_{seed}" / "folds").glob("fold_*.json"))
            if len(fold_paths) != 57:
                raise ValueError(f"Expected 57 fold files: {model}/{config}/seed_{seed}")
            signatures = [fold_signature(path) for path in fold_paths]
            if seed not in split_reference:
                split_reference[seed] = signatures
            elif signatures != split_reference[seed]:
                raise RuntimeError(f"Split/candidate seed mismatch: {model}/{config}/seed_{seed}")
            for path in fold_paths:
                fold = json.loads(path.read_text(encoding="utf-8"))
                if len(fold["candidate_summaries"]) != 4 or any(item["status"] != "usable" for item in fold["candidate_summaries"]):
                    all_candidates_usable = False
                primary = (fold["inner_val_j"], fold["inner_val_f1"], fold["inner_val_acc"])
                tied_primary = sum(
                    (item["val_j"], item["val_f1"], item["val_acc"]) == primary
                    for item in fold["candidate_summaries"] if item["status"] == "usable"
                )
                window_rows.append({
                    "model": model,
                    "config": config,
                    "global_seed": seed,
                    "fold_id": fold["fold_id"],
                    "test_pid": fold["test_pid"],
                    "window_ms": fold["selected_window_ms"],
                    "stride_ms": fold["selected_stride_ms"],
                    "primary_score_tie_count": tied_primary,
                })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "nested_task_metrics.csv", index=False, encoding="utf-8-sig")
    summary = metrics.groupby(["model", "config"])[["accuracy", "f1"]].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["model", "config", "accuracy_mean", "accuracy_sd", "accuracy_min", "accuracy_max", "f1_mean", "f1_sd", "f1_min", "f1_max"]
    summary.to_csv(output / "nested_metric_summary.csv", index=False, encoding="utf-8-sig")

    windows = pd.DataFrame(window_rows)
    windows.to_csv(output / "nested_selected_window_by_fold.csv", index=False, encoding="utf-8-sig")
    window_frequency = (
        windows.groupby(["model", "config", "window_ms", "stride_ms"]).size().rename("count").reset_index()
    )
    window_frequency["percent"] = window_frequency["count"] / 285 * 100
    window_frequency.to_csv(output / "nested_window_selection_frequency.csv", index=False, encoding="utf-8-sig")

    y_true = np.asarray(reference_truth, dtype=int)
    comparison_rows = []
    offset = 0
    comparison_rows.append(paired_row(
        comparison="AttentionNet Full_Preprocessing minus No_Filtering (nested)",
        y_true=y_true,
        pred_a=matrix(tasks, "AttentionNet", FULL),
        pred_b=matrix(tasks, "AttentionNet", NO_FILTER),
        seed_offset=offset,
        family_size=1,
    ))
    offset += 1
    for model_a, model_b in combinations(MODELS, 2):
        comparison_rows.append(paired_row(
            comparison=f"{model_a} minus {model_b} (nested Full_Preprocessing)",
            y_true=y_true,
            pred_a=matrix(tasks, model_a, FULL),
            pred_b=matrix(tasks, model_b, FULL),
            seed_offset=offset,
            family_size=3,
        ))
        offset += 1
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output / "nested_paired_comparisons.csv", index=False, encoding="utf-8-sig")

    fixed_tasks = {}
    for model in MODELS:
        for seed in SEEDS:
            fixed_tasks[(model, FULL, seed)] = load_predictions(fixed_root, model, FULL, seed)
    for seed in SEEDS:
        fixed_tasks[("AttentionNet", NO_FILTER, seed)] = load_predictions(fixed_root, "AttentionNet", NO_FILTER, seed)
    for key, frame in fixed_tasks.items():
        if frame["test_pid"].astype(int).tolist() != reference_pid or frame["true_label"].astype(int).tolist() != reference_truth:
            raise ValueError(f"Fixed/nested participant alignment mismatch: {key}")

    fixed_nested_rows = []
    for model in MODELS:
        nested_matrix = matrix(tasks, model, FULL)
        fixed_matrix = matrix(fixed_tasks, model, FULL)
        fixed_nested_rows.append(paired_row(
            comparison=f"{model} nested minus fixed (Full_Preprocessing)",
            y_true=y_true,
            pred_a=nested_matrix,
            pred_b=fixed_matrix,
            seed_offset=offset,
            family_size=3,
        ))
        offset += 1
    fixed_attention_row = paired_row(
        comparison="AttentionNet Full_Preprocessing minus No_Filtering (fixed reference)",
        y_true=y_true,
        pred_a=matrix(fixed_tasks, "AttentionNet", FULL),
        pred_b=matrix(fixed_tasks, "AttentionNet", NO_FILTER),
        seed_offset=offset,
        family_size=1,
    )
    fixed_nested = pd.DataFrame(fixed_nested_rows + [fixed_attention_row])
    fixed_nested.to_csv(output / "fixed_nested_reference_comparisons.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "status": "complete",
        "n_tasks": len(metrics),
        "n_outer_folds": len(metrics) * 57,
        "n_candidate_trainings": len(metrics) * 57 * 4,
        "n_participants": len(reference_pid),
        "n_asd": int(sum(reference_truth)),
        "n_td": int(len(reference_truth) - sum(reference_truth)),
        "global_seeds": list(SEEDS),
        "independent_unit": "participant",
        "pipeline_seeds_are_repeated_computational_runs": True,
        "no_best_seed_selection": True,
        "all_20_tasks_complete": True,
        "all_candidates_usable": all_candidates_usable,
        "cross_task_split_and_candidate_seed_mismatches": 0,
        "candidate_windows_ms": [list(item) for item in CANDIDATES],
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "scheme": "crossed resampling of pipeline seeds and participants",
            "primary_interval": "two-sided percentile 95% CI",
            "model_family_interval": "Bonferroni simultaneous 98.333% CI for three pairwise comparisons",
        },
        "interpretation_rule": "A numerical difference is not described as statistically supported when its uncertainty interval includes zero.",
    }
    (output / "nested_analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Validated {len(metrics)}/20 tasks, {len(metrics) * 57}/1140 folds, all candidates usable={all_candidates_usable}")
    print("\nMetric summary:")
    print(summary.to_string(index=False))
    print("\nNested paired comparisons:")
    print(comparisons[["comparison", "delta_accuracy_mean", "delta_accuracy_sd_across_seeds", "seeds_a_higher_accuracy", "seeds_equal_accuracy", "seeds_a_lower_accuracy", "accuracy_ci95_low", "accuracy_ci95_high"]].to_string(index=False))
    print("\nFixed/nested reference comparisons:")
    print(fixed_nested[["comparison", "delta_accuracy_mean", "delta_accuracy_sd_across_seeds", "seeds_a_higher_accuracy", "seeds_equal_accuracy", "seeds_a_lower_accuracy", "accuracy_ci95_low", "accuracy_ci95_high"]].to_string(index=False))
    print("\nWindow frequency:")
    print(window_frequency.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
