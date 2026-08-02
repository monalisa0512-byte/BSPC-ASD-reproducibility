#!/usr/bin/env python3
"""Strict fixed-window LOSO task with fold-level seeding and checkpoints.

One process runs exactly one (model, preprocessing condition, global seed)
combination.  Every LOSO fold is independently reseeded, saved atomically, and
can be resumed without recomputing completed folds.
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
import sys
import time
import traceback

# This variable must be set before importing torch for deterministic CUDA GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


MODELS = {
    "AttentionNet": (
        "run_attentionnet_preprocessing_ablation_pipeline_fixed",
        "AttentionNet",
    ),
    "CNNLSTM": (
        "run_cnnlstm_preprocessing_ablation_pipeline_fixed",
        "CNNLSTM",
    ),
    "PureLSTM": (
        "run_purelstm_preprocessing_ablation_pipeline_fixed",
        "PureLSTM",
    ),
}

CONFIGS = (
    "Full_Preprocessing",
    "Linear_Interpolation",
    "Without_Blink_Expansion",
    "No_Filtering",
    "Without_Mask_Features",
)

FOLD_SEED_SCHEME = "global_seed_times_100000_plus_one_based_fold_id_v1"
EXPECTED_PARTICIPANTS = 57


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


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def fold_seed(global_seed: int, fold_id: int) -> int:
    if fold_id < 1:
        raise ValueError("fold_id must be one-based and positive")
    return global_seed * 100_000 + fold_id


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def task_directory(output_root: Path, model: str, config: str, seed: int) -> Path:
    return output_root / model / config / f"seed_{seed}"


def update_progress(progress_path: Path, base: dict, **updates: object) -> dict:
    progress = {**base, **updates, "updated_at": now()}
    atomic_json(progress_path, progress)
    return progress


def load_completed_fold(path: Path, contract_hash: str, fold_id: int, test_pid: int) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_hash") != contract_hash:
        raise RuntimeError(f"Checkpoint contract mismatch: {path}")
    if int(value.get("fold_id", -1)) != fold_id or int(value.get("test_pid", -1)) != int(test_pid):
        raise RuntimeError(f"Checkpoint fold/PID mismatch: {path}")
    if value.get("status") != "complete":
        return None
    return value


def rebuild_prediction_csv(task_dir: Path, contract_hash: str) -> pd.DataFrame:
    rows = []
    for path in sorted((task_dir / "folds").glob("fold_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or value.get("contract_hash") != contract_hash:
            continue
        rows.append(
            {
                "fold_id": value["fold_id"],
                "test_pid": value["test_pid"],
                "true_label": value["true_label"],
                "pred_label": value["pred_label"],
                "median_prob": value["median_prob"],
                "threshold": value["threshold"],
                "n_test_windows": value["n_test_windows"],
                "correct": value["correct"],
                "global_seed": value["global_seed"],
                "fold_seed": value["fold_seed"],
                "epochs_trained": value["epochs_trained"],
                "best_val_j": value["best_val_j"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("fold_id") if rows else pd.DataFrame()
    atomic_csv(task_dir / "fold_predictions.csv", frame)
    return frame


def build_contract(args: argparse.Namespace, module, config, runner_path: Path) -> dict:
    source_path = Path(module.__file__).resolve()
    return {
        "schema_version": 1,
        "model": args.model,
        "config": args.config,
        "global_seed": args.global_seed,
        "fold_seed_scheme": FOLD_SEED_SCHEME,
        "fold_seed_examples": {
            "fold_1": fold_seed(args.global_seed, 1),
            "fold_57": fold_seed(args.global_seed, 57),
        },
        "window_ms": 1000,
        "stride_ms": 500,
        "epochs": args.epochs,
        "batch_size": module.BATCH_SIZE,
        "learning_rate": module.LR,
        "weight_decay": 1e-4,
        "early_stopping_patience": 12,
        "participant_validation_fraction": 0.15,
        "participant_order": "sorted_pid_ascending",
        "same_condition_for_train_validation_test": True,
        "preprocessing": {
            "pre_ms": config.pre_ms,
            "post_ms": config.post_ms,
            "impute_method": config.impute_method,
            "use_mask_features": config.use_mask_features,
            "trial_intercept": config.trial_intercept,
            "window_intercept": config.window_intercept,
            "max_trial_missing_rate": config.max_trial_missing_rate,
            "max_missing_rate": config.max_missing_rate,
            "max_continuous_frames": (
                "inf" if config.max_continuous_frames == float("inf") else config.max_continuous_frames
            ),
        },
        "random_processes_reset_each_fold": [
            "Python random",
            "NumPy",
            "PyTorch CPU",
            "PyTorch CUDA all devices",
            "participant-level train/validation split",
            "weighted sampler",
            "mini-batch order",
            "weight initialization",
            "dropout",
            "optimization trajectory",
            "AMP gradient scaler",
        ],
        "determinism": {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "runner_source": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "model_source": str(source_path),
        "model_source_sha256": sha256_file(source_path),
    }


def run_fold(
    *,
    args: argparse.Namespace,
    module,
    model_class,
    config,
    full_df: pd.DataFrame,
    pid_to_label: dict,
    all_pids: list,
    fold_id: int,
    test_pid: int,
    progress_path: Path,
    progress_base: dict,
) -> dict:
    current_seed = fold_seed(args.global_seed, fold_id)
    seed_everything(current_seed)

    pool_pids = [pid for pid in all_pids if pid != test_pid]
    pool_labels = [pid_to_label[pid] for pid in pool_pids]
    real_train_pids, val_pids = module.get_stratified_train_val_pids(
        pool_pids,
        pool_labels,
        test_size=0.15,
        random_state=current_seed,
    )

    update_progress(
        progress_path,
        progress_base,
        status="running",
        stage="building_fold_data",
        current_fold=fold_id,
        current_test_pid=int(test_pid),
        current_epoch=0,
        fold_seed=current_seed,
    )

    X_tr, Y_tr, P_tr, X_te, Y_te, scaler = module.create_dataset_from_full_data(
        full_df, real_train_pids, test_pid, config, scaler=None
    )
    if X_tr is None or len(X_tr) == 0:
        raise RuntimeError(f"Fold {fold_id}: empty training set")
    if X_te is None or len(X_te) == 0:
        raise RuntimeError(f"Fold {fold_id}: empty test set")

    X_val_list = []
    val_window_counts = {}
    for val_pid in val_pids:
        x_val, y_val = module.extract_single_pid_data(full_df, val_pid, scaler, config)
        if x_val is not None and len(x_val) > 0:
            label = int(np.round(np.mean(y_val)))
            X_val_list.append((torch.tensor(x_val, dtype=torch.float32).to(module.DEVICE), label))
            val_window_counts[str(int(val_pid))] = int(len(x_val))
    if not X_val_list:
        raise RuntimeError(f"Fold {fold_id}: empty validation set")

    # Reset the training substream after condition-dependent deterministic data
    # construction. This guarantees identical training RNG entry state across
    # the five preprocessing conditions for the same global seed and fold.
    seed_everything(current_seed)

    pid_counts = module.Counter(P_tr)
    class_0_pids = set(pid for pid, label in zip(P_tr, Y_tr) if label == 0)
    class_1_pids = set(pid for pid, label in zip(P_tr, Y_tr) if label == 1)
    n_class_0 = max(1, len(class_0_pids))
    n_class_1 = max(1, len(class_1_pids))
    sample_weights = [
        1.0 / (pid_counts[pid] * (n_class_1 if label == 1 else n_class_0))
        for pid, label in zip(P_tr, Y_tr)
    ]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(current_seed)
    sampler = WeightedRandomSampler(
        sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
        generator=sampler_generator,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(current_seed)

    t_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(module.DEVICE)
    t_Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(module.DEVICE)
    t_X_te = torch.tensor(X_te, dtype=torch.float32).to(module.DEVICE)
    train_loader = DataLoader(
        TensorDataset(t_X_tr, t_Y_tr),
        batch_size=module.BATCH_SIZE,
        sampler=sampler,
        generator=loader_generator,
        num_workers=0,
    )

    input_dim = 14 if config.use_mask_features else 7
    model = model_class(input_dim=input_dim).to(module.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=module.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=module.LR,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.cuda.amp.GradScaler()

    best_val_metric = -1.0
    best_threshold = 0.5
    best_weights = None
    patience = 12
    patience_counter = 0
    epochs_trained = 0

    for epoch_index in range(args.epochs):
        epochs_trained = epoch_index + 1
        update_progress(
            progress_path,
            progress_base,
            status="running",
            stage="training",
            current_fold=fold_id,
            current_test_pid=int(test_pid),
            current_epoch=epochs_trained,
            max_epochs=args.epochs,
            fold_seed=current_seed,
            best_val_j=best_val_metric,
        )
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = criterion(model(batch_x), batch_y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

        model.eval()
        val_probabilities = []
        val_labels = []
        with torch.no_grad():
            for tensor_x, label in X_val_list:
                with torch.cuda.amp.autocast():
                    probabilities = torch.sigmoid(model(tensor_x)).cpu().numpy()
                val_probabilities.append(float(np.median(probabilities)))
                val_labels.append(label)

        current_best_j = -1.0
        current_best_threshold = 0.5
        for threshold in np.arange(0.30, 0.71, 0.01):
            predictions = [1 if probability > threshold else 0 for probability in val_probabilities]
            tn, fp, fn, tp = confusion_matrix(val_labels, predictions, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn + 1e-7)
            specificity = tn / (tn + fp + 1e-7)
            j_value = sensitivity + specificity - 1
            if j_value > current_best_j:
                current_best_j = float(j_value)
                current_best_threshold = float(threshold)

        if current_best_j > best_val_metric:
            best_val_metric = current_best_j
            best_threshold = current_best_threshold
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_weights is None:
        raise RuntimeError(f"Fold {fold_id}: no valid checkpoint")
    model.load_state_dict(best_weights)
    model.eval()

    update_progress(
        progress_path,
        progress_base,
        status="running",
        stage="evaluating_fold",
        current_fold=fold_id,
        current_test_pid=int(test_pid),
        current_epoch=epochs_trained,
        fold_seed=current_seed,
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            test_probabilities = torch.sigmoid(model(t_X_te)).cpu().numpy()
    median_probability = float(np.median(test_probabilities))
    true_label = int(Y_te[0])
    predicted_label = 1 if median_probability > best_threshold else 0

    result = {
        "status": "complete",
        "fold_id": fold_id,
        "test_pid": int(test_pid),
        "true_label": true_label,
        "pred_label": predicted_label,
        "correct": int(true_label == predicted_label),
        "median_prob": median_probability,
        "threshold": float(best_threshold),
        "best_val_j": float(best_val_metric),
        "epochs_trained": epochs_trained,
        "n_train_windows": int(len(X_tr)),
        "n_test_windows": int(len(X_te)),
        "validation_window_counts": val_window_counts,
        "train_pids": [int(pid) for pid in real_train_pids],
        "val_pids": [int(pid) for pid in val_pids],
        "global_seed": args.global_seed,
        "fold_seed": current_seed,
        "fold_seed_scheme": FOLD_SEED_SCHEME,
        "completed_at": now(),
    }

    del model, optimizer, scheduler, train_loader, t_X_tr, t_Y_tr, t_X_te
    del X_val_list, X_tr, Y_tr, P_tr, X_te, Y_te
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    configure_determinism()
    data_folder = Path(args.data_folder).resolve()
    output_root = Path(args.output_root).resolve()
    task_dir = task_directory(output_root, args.model, args.config, args.global_seed)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "folds").mkdir(exist_ok=True)

    os.environ["BSPC_FIXED_RESULTS_FOLDER"] = str(task_dir)
    module_name, class_name = MODELS[args.model]
    module = importlib.import_module(module_name)
    module.SEED = args.global_seed
    module.WINDOW_SIZE = 1000
    module.STRIDE = 500
    module.EPOCHS = args.epochs
    module.RESULTS_FOLDER = str(task_dir)
    module.LOG_FILE = str(task_dir / "task_module.log")
    model_class = getattr(module, class_name)
    config_by_name = {item.name: item for item in module.ABLATION_CONFIGS}
    config = config_by_name[args.config]

    runner_path = Path(__file__).resolve()
    contract = build_contract(args, module, config, runner_path)
    contract_hash = canonical_hash(contract)
    contract["contract_hash"] = contract_hash
    contract_path = task_dir / "task_contract.json"
    if contract_path.exists():
        old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if old_contract.get("contract_hash") != contract_hash:
            raise RuntimeError(f"Existing task contract differs: {contract_path}")
    else:
        atomic_json(contract_path, contract)

    if args.contract_only:
        atomic_text(task_dir / "CONTRACT_READY", now() + "\n")
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return 0

    complete_path = task_dir / "COMPLETE"
    if complete_path.exists():
        print(f"Task already complete: {task_dir}")
        return 0
    if not data_folder.is_dir():
        raise FileNotFoundError(data_folder)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a formal strict task; no GPU is available")

    # A failed/interrupted task is explicitly resumable. Clear stale status
    # markers only after the contract and CUDA checks have passed.
    (task_dir / "failure.json").unlink(missing_ok=True)
    (task_dir / "PARTIAL_TEST_COMPLETE").unlink(missing_ok=True)

    progress_path = task_dir / "progress.json"
    progress_base = {
        "schema_version": 1,
        "task_key": f"{args.model}__{args.config}__seed_{args.global_seed}",
        "model": args.model,
        "config": args.config,
        "global_seed": args.global_seed,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "contract_hash": contract_hash,
        "total_folds": EXPECTED_PARTICIPANTS,
        "max_epochs": args.epochs,
        "started_at": now(),
    }
    update_progress(
        progress_path,
        progress_base,
        status="running",
        stage="preprocessing_task_data",
        completed_folds=len(list((task_dir / "folds").glob("fold_*.json"))),
        current_fold=None,
        current_epoch=0,
    )

    full_df = module.load_all_data(str(data_folder), config, is_train_data=True)
    pid_to_label = {
        pid: (1 if str(group[module.Y_COL].iloc[0]).strip().upper() == "ASD" else 0)
        for pid, group in full_df.groupby(module.PID_COL)
    }
    all_pids = sorted(pid_to_label)
    if len(all_pids) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(f"Expected {EXPECTED_PARTICIPANTS} participants, found {len(all_pids)}")

    fold_limit = len(all_pids) if args.max_folds is None else min(args.max_folds, len(all_pids))
    for zero_index, test_pid in enumerate(all_pids[:fold_limit]):
        fold_id = zero_index + 1
        fold_path = task_dir / "folds" / f"fold_{fold_id:03d}.json"
        existing = load_completed_fold(fold_path, contract_hash, fold_id, int(test_pid))
        if existing is not None:
            continue
        fold_result = run_fold(
            args=args,
            module=module,
            model_class=model_class,
            config=config,
            full_df=full_df,
            pid_to_label=pid_to_label,
            all_pids=all_pids,
            fold_id=fold_id,
            test_pid=test_pid,
            progress_path=progress_path,
            progress_base=progress_base,
        )
        fold_result["contract_hash"] = contract_hash
        atomic_json(fold_path, fold_result)
        predictions = rebuild_prediction_csv(task_dir, contract_hash)
        update_progress(
            progress_path,
            progress_base,
            status="running",
            stage="fold_checkpoint_saved",
            completed_folds=len(predictions),
            current_fold=fold_id,
            current_test_pid=int(test_pid),
            current_epoch=fold_result["epochs_trained"],
            fold_seed=fold_result["fold_seed"],
        )

    predictions = rebuild_prediction_csv(task_dir, contract_hash)
    if args.max_folds is not None and args.max_folds < len(all_pids):
        update_progress(
            progress_path,
            progress_base,
            status="partial_test_complete",
            stage="partial_test_complete",
            completed_folds=len(predictions),
            current_fold=None,
            current_epoch=0,
        )
        atomic_text(task_dir / "PARTIAL_TEST_COMPLETE", now() + "\n")
        return 0

    if len(predictions) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(f"Task ended with {len(predictions)} completed folds, expected 57")
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["pred_label"].to_numpy(dtype=int)
    summary = {
        "status": "complete",
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
    update_progress(
        progress_path,
        progress_base,
        status="complete",
        stage="complete",
        completed_folds=EXPECTED_PARTICIPANTS,
        current_fold=None,
        current_epoch=0,
        result=summary,
    )
    atomic_text(complete_path, now() + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # The task directory may not yet be available if argument parsing fails.
        try:
            parsed = parse_args()
            directory = task_directory(
                Path(parsed.output_root).resolve(), parsed.model, parsed.config, parsed.global_seed
            )
            failure = {
                "status": "failed",
                "failed_at": now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "pid": os.getpid(),
            }
            atomic_json(directory / "failure.json", failure)
            progress_file = directory / "progress.json"
            if progress_file.exists():
                progress = json.loads(progress_file.read_text(encoding="utf-8"))
                progress.update(failure)
                atomic_json(progress_file, progress)
        except Exception:
            pass
        raise
