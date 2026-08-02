#!/usr/bin/env python3
"""Report task/fold/epoch progress for the strict 75-task experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import pandas as pd


MODELS = ("AttentionNet", "CNNLSTM", "PureLSTM")
CONFIGS = (
    "Full_Preprocessing",
    "Linear_Interpolation",
    "Without_Blink_Expansion",
    "No_Filtering",
    "Without_Mask_Features",
)
SEEDS = (42, 43, 44, 45, 46)
TOTAL_FOLDS = 57


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--write-snapshot", action="store_true")
    return parser.parse_args()


def process_is_task(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        try:
            return "run_fixed_multiseed_fold_strict.py" in proc_cmdline.read_bytes().decode(
                "utf-8", errors="ignore"
            )
        except OSError:
            return False
    return True


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).resolve()
    rows = []
    split_registry: dict[tuple[str, int, int], dict[str, tuple]] = {}

    for model in MODELS:
        for config in CONFIGS:
            for seed in SEEDS:
                task_dir = root / model / config / f"seed_{seed}"
                progress = read_json(task_dir / "progress.json")
                complete_folds = len(list((task_dir / "folds").glob("fold_*.json")))
                pid = int(progress.get("pid", 0) or 0)
                if (task_dir / "COMPLETE").exists():
                    status = "complete"
                elif (task_dir / "failure.json").exists():
                    status = "failed"
                elif progress and process_is_task(pid):
                    status = "running"
                elif progress or complete_folds:
                    status = "interrupted_resumable"
                else:
                    status = "queued"

                rows.append(
                    {
                        "model": model,
                        "config": config,
                        "global_seed": seed,
                        "status": status,
                        "completed_folds": complete_folds,
                        "total_folds": TOTAL_FOLDS,
                        "current_fold": progress.get("current_fold"),
                        "current_test_pid": progress.get("current_test_pid"),
                        "stage": progress.get("stage"),
                        "current_epoch": progress.get("current_epoch"),
                        "max_epochs": progress.get("max_epochs", 50),
                        "updated_at": progress.get("updated_at"),
                        "pid": pid or None,
                    }
                )

                for fold_path in (task_dir / "folds").glob("fold_*.json"):
                    fold = read_json(fold_path)
                    if fold.get("status") != "complete":
                        continue
                    key = (model, seed, int(fold["fold_id"]))
                    split_registry.setdefault(key, {})[config] = (
                        tuple(fold.get("train_pids", [])),
                        tuple(fold.get("val_pids", [])),
                        int(fold.get("test_pid")),
                        int(fold.get("fold_seed")),
                    )

    frame = pd.DataFrame(rows)
    status_counts = frame["status"].value_counts().to_dict()
    split_mismatches = []
    for key, condition_splits in split_registry.items():
        distinct = set(condition_splits.values())
        if len(distinct) > 1:
            split_mismatches.append(
                {
                    "model": key[0],
                    "global_seed": key[1],
                    "fold_id": key[2],
                    "conditions_present": sorted(condition_splits),
                }
            )

    snapshot = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "output_root": str(root),
        "total_tasks": len(frame),
        "total_expected_folds": len(frame) * TOTAL_FOLDS,
        "completed_folds": int(frame["completed_folds"].sum()),
        "status_counts": status_counts,
        "split_consistency_mismatches": split_mismatches,
        "tasks": rows,
    }

    print(
        f"Tasks: complete={status_counts.get('complete', 0)}, "
        f"running={status_counts.get('running', 0)}, "
        f"resumable={status_counts.get('interrupted_resumable', 0)}, "
        f"failed={status_counts.get('failed', 0)}, "
        f"queued={status_counts.get('queued', 0)}"
    )
    print(
        f"Folds: {snapshot['completed_folds']}/{snapshot['total_expected_folds']} "
        f"({100 * snapshot['completed_folds'] / snapshot['total_expected_folds']:.2f}%)"
    )
    print(f"Cross-condition split mismatches: {len(split_mismatches)}")

    active = frame[frame["status"].isin(["running", "interrupted_resumable", "failed"])]
    if not active.empty:
        print("\nActive/resumable/failed tasks:")
        print(
            active[
                [
                    "model",
                    "config",
                    "global_seed",
                    "status",
                    "completed_folds",
                    "current_fold",
                    "stage",
                    "current_epoch",
                    "max_epochs",
                    "updated_at",
                ]
            ].to_string(index=False)
        )

    if args.write_snapshot:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "progress_snapshot.json", snapshot)
        frame.to_csv(root / "progress_tasks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
