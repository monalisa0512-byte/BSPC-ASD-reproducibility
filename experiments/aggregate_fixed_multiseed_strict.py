#!/usr/bin/env python3
"""Aggregate the strict 75-task experiment without selecting a best seed."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


MODELS = ("AttentionNet", "CNNLSTM", "PureLSTM")
CONFIGS = (
    "Full_Preprocessing",
    "Linear_Interpolation",
    "Without_Blink_Expansion",
    "No_Filtering",
    "Without_Mask_Features",
)
ABLATIONS = CONFIGS[1:]
SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260730


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return float(min(1.0, 2 * tail))


def load_task(root: Path, model: str, config: str, seed: int) -> pd.DataFrame:
    task = root / model / config / f"seed_{seed}"
    if not (task / "COMPLETE").exists():
        raise FileNotFoundError(f"Incomplete task: {task}")
    frame = pd.read_csv(task / "fold_predictions.csv", encoding="utf-8-sig")
    if len(frame) != 57 or frame["test_pid"].nunique() != 57:
        raise ValueError(f"Expected 57 unique participants: {task}")
    return frame.sort_values("fold_id").reset_index(drop=True)


def f1_rows(y_true: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    # y_true: B x 1 x N; prediction: B x S x N
    tp = np.sum((y_true == 1) & (prediction == 1), axis=2)
    fp = np.sum((y_true == 0) & (prediction == 1), axis=2)
    fn = np.sum((y_true == 1) & (prediction == 0), axis=2)
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )


def crossed_seed_participant_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    seed_offset: int,
) -> dict[str, float]:
    # pred_a/pred_b are seed x participant. Seeds and participants are crossed
    # sources of variation; neither 5*57 observations nor windows are treated
    # as independent participants.
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    n_seeds, n_participants = pred_a.shape
    sampled_seeds = rng.integers(0, n_seeds, size=(BOOTSTRAP_ITERATIONS, n_seeds))
    sampled_participants = rng.integers(
        0, n_participants, size=(BOOTSTRAP_ITERATIONS, n_participants)
    )
    pa = pred_a[sampled_seeds[:, :, None], sampled_participants[:, None, :]]
    pb = pred_b[sampled_seeds[:, :, None], sampled_participants[:, None, :]]
    yt = y_true[sampled_participants][:, None, :]
    acc_delta = np.mean(yt == pa, axis=(1, 2)) - np.mean(yt == pb, axis=(1, 2))
    f1_delta = np.mean(f1_rows(yt, pa), axis=1) - np.mean(f1_rows(yt, pb), axis=1)
    acc_low, acc_high = np.percentile(acc_delta, [2.5, 97.5])
    f1_low, f1_high = np.percentile(f1_delta, [2.5, 97.5])
    return {
        "crossed_bootstrap_delta_accuracy_ci_low": float(acc_low),
        "crossed_bootstrap_delta_accuracy_ci_high": float(acc_high),
        "crossed_bootstrap_delta_f1_ci_low": float(f1_low),
        "crossed_bootstrap_delta_f1_ci_high": float(f1_high),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    tasks: dict[tuple[str, str, int], pd.DataFrame] = {}
    metric_rows = []
    reference_pid = None
    reference_truth = None
    split_mismatches = []

    for model in MODELS:
        for seed in SEEDS:
            split_by_condition = {}
            for config in CONFIGS:
                frame = load_task(root, model, config, seed)
                pid = frame["test_pid"].astype(int).tolist()
                truth = frame["true_label"].astype(int).tolist()
                if reference_pid is None:
                    reference_pid, reference_truth = pid, truth
                if pid != reference_pid or truth != reference_truth:
                    raise ValueError(f"Participant alignment mismatch: {model}/{config}/seed_{seed}")
                tasks[(model, config, seed)] = frame
                metric_rows.append(
                    {
                        "model": model,
                        "config": config,
                        "global_seed": seed,
                        "accuracy": accuracy_score(frame["true_label"], frame["pred_label"]),
                        "f1": f1_score(frame["true_label"], frame["pred_label"], zero_division=0),
                    }
                )
                task_dir = root / model / config / f"seed_{seed}" / "folds"
                splits = []
                for fold_path in sorted(task_dir.glob("fold_*.json")):
                    fold = json.loads(fold_path.read_text(encoding="utf-8"))
                    splits.append(
                        (
                            fold["fold_id"],
                            tuple(fold["train_pids"]),
                            tuple(fold["val_pids"]),
                            fold["test_pid"],
                            fold["fold_seed"],
                        )
                    )
                split_by_condition[config] = splits
            first = split_by_condition[CONFIGS[0]]
            for config in CONFIGS[1:]:
                if split_by_condition[config] != first:
                    split_mismatches.append({"model": model, "seed": seed, "config": config})

    if split_mismatches:
        raise RuntimeError(f"Cross-condition split mismatch: {split_mismatches}")

    task_metrics = pd.DataFrame(metric_rows)
    task_metrics.to_csv(output / "multiseed_task_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        task_metrics.groupby(["model", "config"])[["accuracy", "f1"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "model",
        "config",
        "accuracy_mean",
        "accuracy_sd",
        "accuracy_min",
        "accuracy_max",
        "f1_mean",
        "f1_sd",
        "f1_min",
        "f1_max",
    ]
    summary.to_csv(output / "multiseed_metric_summary.csv", index=False, encoding="utf-8-sig")

    paired_rows = []
    y_true = np.asarray(reference_truth, dtype=int)
    offset = 0
    for model in MODELS:
        full_matrix = np.stack(
            [tasks[(model, CONFIGS[0], seed)]["pred_label"].to_numpy(dtype=int) for seed in SEEDS]
        )
        for ablation in ABLATIONS:
            ablation_matrix = np.stack(
                [tasks[(model, ablation, seed)]["pred_label"].to_numpy(dtype=int) for seed in SEEDS]
            )
            seed_delta_accuracy = []
            seed_delta_f1 = []
            seed_mcnemar = []
            for seed_index, seed in enumerate(SEEDS):
                full_pred = full_matrix[seed_index]
                ablation_pred = ablation_matrix[seed_index]
                full_correct = full_pred == y_true
                ablation_correct = ablation_pred == y_true
                b = int(np.sum(full_correct & ~ablation_correct))
                c = int(np.sum(~full_correct & ablation_correct))
                seed_delta_accuracy.append(
                    accuracy_score(y_true, full_pred) - accuracy_score(y_true, ablation_pred)
                )
                seed_delta_f1.append(
                    f1_score(y_true, full_pred, zero_division=0)
                    - f1_score(y_true, ablation_pred, zero_division=0)
                )
                seed_mcnemar.append(exact_mcnemar_p(b, c))
            boot = crossed_seed_participant_bootstrap(
                y_true, full_matrix, ablation_matrix, offset
            )
            offset += 1
            paired_rows.append(
                {
                    "model": model,
                    "comparison": f"Full_Preprocessing vs {ablation}",
                    "n_participants": len(y_true),
                    "n_seeds": len(SEEDS),
                    "delta_accuracy_mean": float(np.mean(seed_delta_accuracy)),
                    "delta_accuracy_sd": float(np.std(seed_delta_accuracy, ddof=1)),
                    "delta_f1_mean": float(np.mean(seed_delta_f1)),
                    "delta_f1_sd": float(np.std(seed_delta_f1, ddof=1)),
                    "seeds_full_accuracy_higher": int(np.sum(np.asarray(seed_delta_accuracy) > 0)),
                    "seeds_equal_accuracy": int(np.sum(np.asarray(seed_delta_accuracy) == 0)),
                    "seed_delta_accuracy": json.dumps(dict(zip(SEEDS, seed_delta_accuracy))),
                    "seed_delta_f1": json.dumps(dict(zip(SEEDS, seed_delta_f1))),
                    "seed_mcnemar_p": json.dumps(dict(zip(SEEDS, seed_mcnemar))),
                    **boot,
                }
            )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "multiseed_paired_differences.csv", index=False, encoding="utf-8-sig")

    model_rows = []
    for model_a, model_b in combinations(MODELS, 2):
        matrix_a = np.stack(
            [tasks[(model_a, CONFIGS[0], seed)]["pred_label"].to_numpy(dtype=int) for seed in SEEDS]
        )
        matrix_b = np.stack(
            [tasks[(model_b, CONFIGS[0], seed)]["pred_label"].to_numpy(dtype=int) for seed in SEEDS]
        )
        seed_delta_accuracy = []
        seed_delta_f1 = []
        seed_mcnemar = []
        for seed_index, seed in enumerate(SEEDS):
            pred_a = matrix_a[seed_index]
            pred_b = matrix_b[seed_index]
            correct_a = pred_a == y_true
            correct_b = pred_b == y_true
            b = int(np.sum(correct_a & ~correct_b))
            c = int(np.sum(~correct_a & correct_b))
            seed_delta_accuracy.append(
                accuracy_score(y_true, pred_a) - accuracy_score(y_true, pred_b)
            )
            seed_delta_f1.append(
                f1_score(y_true, pred_a, zero_division=0)
                - f1_score(y_true, pred_b, zero_division=0)
            )
            seed_mcnemar.append(exact_mcnemar_p(b, c))
        boot = crossed_seed_participant_bootstrap(y_true, matrix_a, matrix_b, offset)
        offset += 1
        model_rows.append(
            {
                "condition": CONFIGS[0],
                "comparison": f"{model_a} vs {model_b}",
                "n_participants": len(y_true),
                "n_seeds": len(SEEDS),
                "delta_accuracy_mean": float(np.mean(seed_delta_accuracy)),
                "delta_accuracy_sd": float(np.std(seed_delta_accuracy, ddof=1)),
                "delta_f1_mean": float(np.mean(seed_delta_f1)),
                "delta_f1_sd": float(np.std(seed_delta_f1, ddof=1)),
                "seed_delta_accuracy": json.dumps(dict(zip(SEEDS, seed_delta_accuracy))),
                "seed_delta_f1": json.dumps(dict(zip(SEEDS, seed_delta_f1))),
                "seed_mcnemar_p": json.dumps(dict(zip(SEEDS, seed_mcnemar))),
                **boot,
            }
        )
    model_paired = pd.DataFrame(model_rows)
    model_paired.to_csv(
        output / "multiseed_full_model_paired_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "status": "complete",
        "independent_unit": "participant",
        "n_participants": len(reference_pid),
        "n_asd": int(sum(reference_truth)),
        "n_td": int(len(reference_truth) - sum(reference_truth)),
        "global_seeds": list(SEEDS),
        "n_tasks": len(task_metrics),
        "fold_seed_reset": True,
        "cross_condition_split_mismatches": split_mismatches,
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "scheme": "crossed resampling of global seeds and participants",
        },
        "model_comparisons": "All three pairwise model comparisons under Full Preprocessing",
        "interpretation": (
            "Global seeds are repeated computational runs, not additional independent participants. "
            "No best-seed selection is performed."
        ),
    }
    (output / "multiseed_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Validated and aggregated {len(task_metrics)}/75 tasks")
    print(summary.to_string(index=False))
    print("\nPaired Full-minus-ablation differences:")
    print(
        paired[
            [
                "model",
                "comparison",
                "delta_accuracy_mean",
                "delta_accuracy_sd",
                "delta_f1_mean",
                "delta_f1_sd",
                "seeds_full_accuracy_higher",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
