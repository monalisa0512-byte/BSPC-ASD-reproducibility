#!/usr/bin/env python3
"""Strict nested-window LOSO task with fold/candidate checkpoints.

One process runs exactly one (model, preprocessing condition, global seed)
combination. Window selection is performed independently inside each outer LOSO
fold using only development participants. The outer test participant is never
used for candidate selection, threshold selection, or standardization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import run_fixed_multiseed_fold_strict as fixed


MODELS = fixed.MODELS
CONFIGS = fixed.CONFIGS
CANDIDATES = ((2000, 1000), (1500, 750), (1000, 500), (2000, 500))
EXPECTED_PARTICIPANTS = 57
PROTOCOL = "nested_window_selection_strict_fold_seed_v1"
CANDIDATE_SEED_SCHEME = "fold_seed_times_10_plus_one_based_candidate_index_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--global-seed", required=True, type=int, choices=range(42, 47))
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def candidate_seed(fold_seed: int, candidate_index: int) -> int:
    if candidate_index < 1 or candidate_index > len(CANDIDATES):
        raise ValueError("candidate_index must be one-based and within CANDIDATES")
    return fold_seed * 10 + candidate_index


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configure_determinism() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def task_directory(root: Path, model: str, config: str, seed: int) -> Path:
    return root / model / config / f"seed_{seed}"


def progress_update(path: Path, base: dict, **updates: object) -> None:
    fixed.atomic_json(path, {**base, **updates, "updated_at": now()})


def config_dict(config) -> dict:
    continuous = config.max_continuous_frames
    return {
        "pre_ms": config.pre_ms,
        "post_ms": config.post_ms,
        "impute_method": config.impute_method,
        "use_mask_features": config.use_mask_features,
        "trial_intercept": config.trial_intercept,
        "window_intercept": config.window_intercept,
        "max_trial_missing_rate": config.max_trial_missing_rate,
        "max_missing_rate": config.max_missing_rate,
        "max_continuous_frames": "inf" if continuous == float("inf") else continuous,
    }


def build_contract(args: argparse.Namespace, module, config, runner_path: Path) -> dict:
    source = Path(module.__file__).resolve()
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "model": args.model,
        "config": args.config,
        "global_seed": args.global_seed,
        "outer_fold_seed_scheme": fixed.FOLD_SEED_SCHEME,
        "candidate_seed_scheme": CANDIDATE_SEED_SCHEME,
        "candidates_ms": [list(pair) for pair in CANDIDATES],
        "selection_score": ["validation_youden_j", "validation_f1", "validation_accuracy", "negative_window_ms", "negative_stride_ms"],
        "window_selection_uses_outer_test": False,
        "standardization_uses_outer_test": False,
        "window_size_fixed_reference_ms": [1000, 500],
        "epochs": args.epochs,
        "batch_size": module.BATCH_SIZE,
        "learning_rate": module.LR,
        "early_stopping_patience": 12,
        "participant_validation_fraction": 0.15,
        "preprocessing": config_dict(config),
        "same_condition_train_validation_test": True,
        "random_processes_reset": [
            "Python random",
            "NumPy",
            "PyTorch CPU/CUDA",
            "outer train/validation participant split",
            "candidate-specific model initialization",
            "candidate-specific sampler and mini-batch order",
            "dropout and optimization trajectory",
            "candidate-specific AMP GradScaler",
        ],
        "runner_source": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "model_source": str(source),
        "model_source_sha256": sha256_file(source),
    }


def load_checkpoint(path: Path, contract_hash: str, fold_id: int, test_pid: int) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_hash") != contract_hash:
        raise RuntimeError(f"Nested checkpoint contract mismatch: {path}")
    if int(value.get("fold_id", -1)) != fold_id or int(value.get("test_pid", -1)) != int(test_pid):
        raise RuntimeError(f"Nested checkpoint fold/PID mismatch: {path}")
    return value if value.get("status") == "complete" else None


def rebuild_predictions(task_dir: Path, contract_hash: str) -> pd.DataFrame:
    rows = []
    for path in sorted((task_dir / "folds").glob("fold_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or value.get("contract_hash") != contract_hash:
            continue
        rows.append({
            "fold_id": value["fold_id"],
            "test_pid": value["test_pid"],
            "true_label": value["true_label"],
            "pred_label": value["pred_label"],
            "median_prob": value["median_prob"],
            "threshold": value["threshold"],
            "selected_window_ms": value["selected_window_ms"],
            "selected_stride_ms": value["selected_stride_ms"],
            "inner_val_j": value["inner_val_j"],
            "inner_val_f1": value["inner_val_f1"],
            "inner_val_acc": value["inner_val_acc"],
            "n_test_windows": value["n_test_windows"],
            "correct": value["correct"],
            "global_seed": value["global_seed"],
            "fold_seed": value["fold_seed"],
        })
    frame = pd.DataFrame(rows).sort_values("fold_id") if rows else pd.DataFrame()
    atomic_csv(task_dir / "fold_predictions.csv", frame)
    return frame


def extract_validation(module, full_df, val_pid, scaler, config, window_ms, stride_ms):
    # The base validation extractor uses module.WINDOW_SIZE/STRIDE, so keep the
    # candidate values explicit and condition-aware.
    old_window, old_stride = module.WINDOW_SIZE, module.STRIDE
    module.WINDOW_SIZE, module.STRIDE = window_ms, stride_ms
    try:
        return module.extract_single_pid_data(full_df, val_pid, scaler, config)
    finally:
        module.WINDOW_SIZE, module.STRIDE = old_window, old_stride


def train_candidate(
    *, args, module, model_class, full_df, train_pids, val_pids, test_pid,
    config, window_ms, stride_ms, candidate_index, fold_seed_value, progress_path, progress_base, fold_id,
):
    candidate_rng_seed = candidate_seed(fold_seed_value, candidate_index)
    fixed.seed_everything(candidate_rng_seed)
    module.WINDOW_SIZE, module.STRIDE = window_ms, stride_ms
    progress_update(
        progress_path,
        progress_base,
        status="running",
        stage="candidate_data",
        current_fold=fold_id,
        current_test_pid=int(test_pid),
        candidate_index=candidate_index,
        candidate_window_ms=window_ms,
        candidate_stride_ms=stride_ms,
        current_epoch=0,
        candidate_seed=candidate_rng_seed,
    )
    out = module.create_dataset_from_full_data(full_df, train_pids, test_pid, config, scaler=None)
    if out[0] is None or len(out[0]) == 0 or out[3] is None or len(out[3]) == 0:
        return None
    X_tr, Y_tr, P_tr, X_te, Y_te, scaler = out

    validation_subjects = []
    for val_pid in val_pids:
        X_val, Y_val = extract_validation(module, full_df, val_pid, scaler, config, window_ms, stride_ms)
        if X_val is not None and len(X_val):
            validation_subjects.append((torch.tensor(X_val, dtype=torch.float32).to(module.DEVICE), int(np.round(np.mean(Y_val)))))
    if not validation_subjects:
        return None

    # Reset the candidate training substream after deterministic data building.
    fixed.seed_everything(candidate_rng_seed)
    pid_counts = module.Counter(P_tr)
    class0 = {pid for pid, label in zip(P_tr, Y_tr) if label == 0}
    class1 = {pid for pid, label in zip(P_tr, Y_tr) if label == 1}
    n0, n1 = max(1, len(class0)), max(1, len(class1))
    weights = [1.0 / (pid_counts[pid] * (n1 if label == 1 else n0)) for pid, label in zip(P_tr, Y_tr)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(candidate_rng_seed)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(candidate_rng_seed)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32), len(weights), replacement=True, generator=generator)
    t_X = torch.tensor(X_tr, dtype=torch.float32).to(module.DEVICE)
    t_Y = torch.tensor(Y_tr, dtype=torch.float32).to(module.DEVICE)
    loader = DataLoader(TensorDataset(t_X, t_Y), batch_size=module.BATCH_SIZE, sampler=sampler, generator=loader_generator, num_workers=0)
    input_dim = 14 if config.use_mask_features else 7
    model = model_class(input_dim=input_dim).to(module.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=module.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=module.LR, steps_per_epoch=len(loader), epochs=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.cuda.amp.GradScaler()
    best_j, best_threshold, best_weights = -1.0, 0.5, None
    best_acc, best_f1 = 0.0, 0.0
    patience_counter = 0
    epochs_trained = 0
    for epoch in range(args.epochs):
        epochs_trained = epoch + 1
        progress_update(progress_path, progress_base, status="running", stage="candidate_training", current_fold=fold_id, current_test_pid=int(test_pid), candidate_index=candidate_index, candidate_window_ms=window_ms, candidate_stride_ms=stride_ms, current_epoch=epochs_trained, max_epochs=args.epochs, candidate_seed=candidate_rng_seed, best_val_j=best_j)
        model.train()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = criterion(model(bx), by)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for X_val, label in validation_subjects:
                with torch.cuda.amp.autocast():
                    p = torch.sigmoid(model(X_val)).cpu().numpy()
                probs.append(float(np.median(p)))
                labels.append(label)
        current_j, current_threshold, current_acc, current_f1 = -1.0, 0.5, 0.0, 0.0
        for threshold in np.arange(0.30, 0.71, 0.01):
            pred = [int(p > threshold) for p in probs]
            tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
            j = tp / (tp + fn + 1e-7) + tn / (tn + fp + 1e-7) - 1
            if j > current_j:
                current_j = float(j)
                current_threshold = float(threshold)
                current_acc = float(accuracy_score(labels, pred))
                current_f1 = float(f1_score(labels, pred, zero_division=0))
        if current_j > best_j:
            best_j, best_threshold, best_acc, best_f1 = current_j, current_threshold, current_acc, current_f1
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 12:
            break

    if best_weights is None:
        return None
    result = {
        "window_size_ms": window_ms,
        "stride_ms": stride_ms,
        "candidate_index": candidate_index,
        "candidate_seed": candidate_rng_seed,
        "weights": best_weights,
        "input_dim": input_dim,
        "X_te": X_te,
        "Y_te": Y_te,
        "threshold": best_threshold,
        "val_j": best_j,
        "val_acc": best_acc,
        "val_f1": best_f1,
        "train_windows": len(X_tr),
        "test_windows": len(X_te),
        "epochs_trained": epochs_trained,
    }
    del model, optimizer, scheduler, loader, t_X, t_Y, validation_subjects
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_fold(*, args, module, model_class, full_df, pid_to_label, all_pids, config, fold_id, test_pid, progress_path, progress_base):
    fold_seed_value = fixed.fold_seed(args.global_seed, fold_id)
    fixed.seed_everything(fold_seed_value)
    pool = [pid for pid in all_pids if pid != test_pid]
    labels = [pid_to_label[pid] for pid in pool]
    train_pids, val_pids = module.get_stratified_train_val_pids(pool, labels, test_size=0.15, random_state=fold_seed_value)
    candidates = []
    candidate_summaries = []
    selected_score = None
    for index, (window_ms, stride_ms) in enumerate(CANDIDATES, start=1):
        result = train_candidate(args=args, module=module, model_class=model_class, full_df=full_df, train_pids=train_pids, val_pids=val_pids, test_pid=test_pid, config=config, window_ms=window_ms, stride_ms=stride_ms, candidate_index=index, fold_seed_value=fold_seed_value, progress_path=progress_path, progress_base=progress_base, fold_id=fold_id)
        if result is None:
            candidate_summaries.append({"candidate_index": index, "window_size_ms": window_ms, "stride_ms": stride_ms, "status": "unusable"})
            continue
        candidate_score = (result["val_j"], result["val_f1"], result["val_acc"], -window_ms, -stride_ms)
        candidate_summaries.append({"candidate_index": index, "window_size_ms": window_ms, "stride_ms": stride_ms, "candidate_seed": result["candidate_seed"], "status": "usable", "val_j": result["val_j"], "val_f1": result["val_f1"], "val_acc": result["val_acc"], "epochs_trained": result["epochs_trained"], "train_windows": result["train_windows"], "test_windows": result["test_windows"]})
        if selected_score is None or candidate_score > selected_score:
            if candidates:
                del candidates[0]
            candidates = [result]
            selected_score = candidate_score
        else:
            del result["weights"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not candidates:
        raise RuntimeError(f"No usable nested candidate at fold {fold_id}")
    selected = candidates[0]
    fixed.seed_everything(selected["candidate_seed"])
    model = model_class(input_dim=selected["input_dim"]).to(module.DEVICE)
    model.load_state_dict(selected["weights"])
    model.eval()
    t_X_te = torch.tensor(selected["X_te"], dtype=torch.float32).to(module.DEVICE)
    progress_update(progress_path, progress_base, status="running", stage="evaluating_selected_window", current_fold=fold_id, current_test_pid=int(test_pid), candidate_index=selected["candidate_index"], candidate_window_ms=selected["window_size_ms"], candidate_stride_ms=selected["stride_ms"], current_epoch=selected["epochs_trained"], fold_seed=fold_seed_value)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            p = torch.sigmoid(model(t_X_te)).cpu().numpy()
    median_prob = float(np.median(p))
    true_label = int(selected["Y_te"][0])
    pred_label = int(median_prob > selected["threshold"])
    del model, t_X_te, candidates
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "complete",
        "fold_id": fold_id,
        "test_pid": int(test_pid),
        "true_label": true_label,
        "pred_label": pred_label,
        "correct": int(pred_label == true_label),
        "median_prob": median_prob,
        "threshold": float(selected["threshold"]),
        "selected_window_ms": selected["window_size_ms"],
        "selected_stride_ms": selected["stride_ms"],
        "selected_candidate_index": selected["candidate_index"],
        "selected_candidate_seed": selected["candidate_seed"],
        "inner_val_j": selected["val_j"],
        "inner_val_f1": selected["val_f1"],
        "inner_val_acc": selected["val_acc"],
        "n_test_windows": selected["test_windows"],
        "candidate_summaries": candidate_summaries,
        "train_pids": [int(pid) for pid in train_pids],
        "val_pids": [int(pid) for pid in val_pids],
        "global_seed": args.global_seed,
        "fold_seed": fold_seed_value,
        "fold_seed_scheme": fixed.FOLD_SEED_SCHEME,
        "candidate_seed_scheme": CANDIDATE_SEED_SCHEME,
        "completed_at": now(),
    }


def main() -> int:
    args = parse_args()
    configure_determinism()
    output_root = Path(args.output_root).resolve()
    task_dir = task_directory(output_root, args.model, args.config, args.global_seed)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "folds").mkdir(exist_ok=True)
    os.environ["BSPC_FIXED_RESULTS_FOLDER"] = str(task_dir)
    module_name, class_name = MODELS[args.model]
    module = importlib.import_module(module_name)
    module.SEED = args.global_seed
    module.RESULTS_FOLDER = str(task_dir)
    module.LOG_FILE = str(task_dir / "task_module.log")
    model_class = getattr(module, class_name)
    config = {item.name: item for item in module.ABLATION_CONFIGS}[args.config]
    runner_path = Path(__file__).resolve()
    contract = build_contract(args, module, config, runner_path)
    contract_hash = canonical_hash(contract)
    contract["contract_hash"] = contract_hash
    contract_path = task_dir / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("contract_hash") != contract_hash:
            raise RuntimeError(f"Existing nested task contract differs: {contract_path}")
    else:
        atomic_json(contract_path, contract)
    if args.contract_only:
        atomic_text(task_dir / "CONTRACT_READY", now() + "\n")
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return 0
    if (task_dir / "COMPLETE").exists():
        print(f"Nested task already complete: {task_dir}")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a formal nested task")
    (task_dir / "failure.json").unlink(missing_ok=True)
    progress_path = task_dir / "progress.json"
    progress_base = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task_key": f"{args.model}__{args.config}__seed_{args.global_seed}",
        "model": args.model,
        "config": args.config,
        "global_seed": args.global_seed,
        "candidate_windows": [list(pair) for pair in CANDIDATES],
        "contract_hash": contract_hash,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "total_folds": EXPECTED_PARTICIPANTS,
        "max_epochs": args.epochs,
        "started_at": now(),
    }
    progress_update(progress_path, progress_base, status="running", stage="preprocessing_task_data", completed_folds=0, current_fold=None, current_epoch=0)
    full_df = module.load_all_data(str(Path(args.data_folder).resolve()), config, is_train_data=True)
    pid_to_label = {pid: (1 if str(group[module.Y_COL].iloc[0]).strip().upper() == "ASD" else 0) for pid, group in full_df.groupby(module.PID_COL)}
    all_pids = sorted(pid_to_label)
    if len(all_pids) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(f"Expected {EXPECTED_PARTICIPANTS} participants, found {len(all_pids)}")
    limit = len(all_pids) if args.max_folds is None else min(args.max_folds, len(all_pids))
    for index, test_pid in enumerate(all_pids[:limit], start=1):
        fold_path = task_dir / "folds" / f"fold_{index:03d}.json"
        existing = load_checkpoint(fold_path, contract_hash, index, int(test_pid))
        if existing is not None:
            continue
        result = run_fold(args=args, module=module, model_class=model_class, full_df=full_df, pid_to_label=pid_to_label, all_pids=all_pids, config=config, fold_id=index, test_pid=test_pid, progress_path=progress_path, progress_base=progress_base)
        result["contract_hash"] = contract_hash
        atomic_json(fold_path, result)
        predictions = rebuild_predictions(task_dir, contract_hash)
        progress_update(progress_path, progress_base, status="running", stage="fold_checkpoint_saved", completed_folds=len(predictions), current_fold=index, current_test_pid=int(test_pid), current_epoch=0)
    predictions = rebuild_predictions(task_dir, contract_hash)
    if args.max_folds is not None and args.max_folds < len(all_pids):
        progress_update(progress_path, progress_base, status="partial_test_complete", stage="partial_test_complete", completed_folds=len(predictions), current_fold=None, current_epoch=0)
        atomic_text(task_dir / "PARTIAL_TEST_COMPLETE", now() + "\n")
        return 0
    if len(predictions) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(f"Nested task ended with {len(predictions)} folds")
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["pred_label"].to_numpy(dtype=int)
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "model": args.model,
        "config": args.config,
        "global_seed": args.global_seed,
        "n_participants": len(predictions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "contract_hash": contract_hash,
        "predictions_sha256": sha256_file(task_dir / "fold_predictions.csv"),
        "completed_at": now(),
    }
    atomic_json(task_dir / "result_summary.json", summary)
    atomic_csv(task_dir / "result_summary.csv", pd.DataFrame([summary]))
    progress_update(progress_path, progress_base, status="complete", stage="complete", completed_folds=EXPECTED_PARTICIPANTS, current_fold=None, current_epoch=0, result=summary)
    atomic_text(task_dir / "COMPLETE", now() + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            args = parse_args()
            directory = task_directory(Path(args.output_root).resolve(), args.model, args.config, args.global_seed)
            failure = {"status": "failed", "failed_at": now(), "error": repr(error), "traceback": traceback.format_exc(), "pid": os.getpid()}
            atomic_json(directory / "failure.json", failure)
            if (directory / "progress.json").exists():
                progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
                progress.update(failure)
                atomic_json(directory / "progress.json", progress)
        except Exception:
            pass
        raise
