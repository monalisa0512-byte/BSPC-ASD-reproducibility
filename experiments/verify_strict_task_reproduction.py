#!/usr/bin/env python3
"""Require exact participant-level reproduction of two completed strict tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def canonical_fold(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("completed_at", None)
    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    for run in (run_a, run_b):
        if not (run / "COMPLETE").exists():
            errors.append(f"Missing COMPLETE marker: {run}")

    contract_a = json.loads((run_a / "task_contract.json").read_text(encoding="utf-8"))
    contract_b = json.loads((run_b / "task_contract.json").read_text(encoding="utf-8"))
    if contract_a != contract_b:
        errors.append("Task contracts differ")

    prediction_a = pd.read_csv(run_a / "fold_predictions.csv", encoding="utf-8-sig")
    prediction_b = pd.read_csv(run_b / "fold_predictions.csv", encoding="utf-8-sig")
    if len(prediction_a) != 57 or len(prediction_b) != 57:
        errors.append(f"Expected 57 folds, got {len(prediction_a)} and {len(prediction_b)}")
    if list(prediction_a.columns) != list(prediction_b.columns):
        errors.append("Prediction columns differ")
    elif not prediction_a.equals(prediction_b):
        different = prediction_a.ne(prediction_b).any(axis=1)
        errors.append(f"Prediction rows differ at folds: {prediction_a.loc[different, 'fold_id'].tolist()}")

    folds_a = sorted((run_a / "folds").glob("fold_*.json"))
    folds_b = sorted((run_b / "folds").glob("fold_*.json"))
    if [path.name for path in folds_a] != [path.name for path in folds_b]:
        errors.append("Fold checkpoint file sets differ")
    else:
        for path_a, path_b in zip(folds_a, folds_b):
            if canonical_fold(path_a) != canonical_fold(path_b):
                errors.append(f"Fold checkpoint differs: {path_a.name}")

    summary_a = json.loads((run_a / "result_summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((run_b / "result_summary.json").read_text(encoding="utf-8"))
    for summary in (summary_a, summary_b):
        summary.pop("completed_at", None)
    if summary_a != summary_b:
        errors.append("Result summaries differ")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "n_folds_a": len(prediction_a),
        "n_folds_b": len(prediction_b),
        "prediction_sha256_a": file_hash(run_a / "fold_predictions.csv"),
        "prediction_sha256_b": file_hash(run_b / "fold_predictions.csv"),
        "contract_hash": contract_a.get("contract_hash"),
        "errors": errors,
    }
    atomic_json(output_dir / "reproducibility_verification.json", report)
    if errors:
        (output_dir / "PASS").unlink(missing_ok=True)
        raise SystemExit("Reproducibility gate failed:\n- " + "\n- ".join(errors))
    (output_dir / "PASS").write_text(
        "Exact 57-fold reproduction verified.\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
