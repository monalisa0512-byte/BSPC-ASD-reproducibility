"""
CNNLSTM candidate-level nested selection diagnostic.

This mirrors run_attentionnet_candidate_selection_diagnostic.py for CNNLSTM.
It records every candidate window's validation metrics and outer-test result,
using a fold-level shared seed for candidate comparability.
"""
import argparse
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from run_model_comparison import (
    BATCH_SIZE,
    CNNLSTM,
    DATA_FOLDER,
    DEVICE,
    EPOCHS,
    LR,
    PID_COL,
    SEED,
    WINDOW_CANDIDATES,
    Y_COL,
    check_mask_distribution,
    create_dataset_from_full_data_pure,
    extract_single_pid_data_pure,
    get_stratified_train_val_pids,
    load_all_data,
)


DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "cnnlstm_candidate_selection_diagnostic")
)


def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_subject_balanced_loader(X_tr, Y_tr, P_tr):
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    t_Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(DEVICE)

    pid_counts = Counter(P_tr)
    class_0_pids = set(p for p, y in zip(P_tr, Y_tr) if y == 0)
    class_1_pids = set(p for p, y in zip(P_tr, Y_tr) if y == 1)
    n_class_0_pids = max(1, len(class_0_pids))
    n_class_1_pids = max(1, len(class_1_pids))

    sample_weights = []
    for p, y in zip(P_tr, Y_tr):
        n_pids_in_class = n_class_1_pids if y == 1 else n_class_0_pids
        sample_weights.append(1.0 / (pid_counts[p] * n_pids_in_class))

    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(TensorDataset(t_X_tr, t_Y_tr), batch_size=BATCH_SIZE, sampler=sampler)
    return loader, t_X_tr, t_Y_tr


def select_threshold_by_youden(probs, labels):
    best_j, best_th, best_preds = -1.0, 0.5, None
    for th in np.arange(0.30, 0.71, 0.01):
        preds = [1 if p > th else 0 for p in probs]
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-7)
        spec = tn / (tn + fp + 1e-7)
        j_stat = sens + spec - 1
        if j_stat > best_j:
            best_j, best_th, best_preds = j_stat, float(th), preds
    return best_j, best_th, accuracy_score(labels, best_preds), f1_score(labels, best_preds)


def evaluate_validation_subjects(model, X_val_list):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for t_X_v, y_v in X_val_list:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                window_probs = torch.sigmoid(model(t_X_v)).detach().cpu().numpy()
            probs.append(float(np.median(window_probs)))
            labels.append(int(y_v))
    return probs, labels


def train_one_candidate(full_df, train_pids, val_pids, test_pid, window_size_ms, stride_ms, fold_seed):
    set_experiment_seed(fold_seed)

    out = create_dataset_from_full_data_pure(
        full_df,
        train_pids,
        test_pid,
        window_size_ms=window_size_ms,
        stride_ms=stride_ms,
    )
    if out[0] is None or len(out[0]) == 0:
        return None

    X_tr, Y_tr, P_tr, X_te, Y_te, scaler = out
    X_val_list = []
    for val_pid in val_pids:
        x_v, y_v = extract_single_pid_data_pure(
            full_df,
            val_pid,
            scaler,
            window_size_ms=window_size_ms,
            stride_ms=stride_ms,
        )
        if x_v is not None and len(x_v) > 0:
            label_val = int(np.round(np.mean(y_v)))
            X_val_list.append((torch.tensor(x_v, dtype=torch.float32).to(DEVICE), label_val))

    if not X_val_list:
        return None

    train_loader, t_X_tr, t_Y_tr = make_subject_balanced_loader(X_tr, Y_tr, P_tr)
    model = CNNLSTM(input_dim=X_tr.shape[2]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_j, best_val_f1, best_val_acc = -1.0, 0.0, 0.0
    best_thresh, best_weights, best_epoch = 0.5, None, -1
    patience, patience_counter = 12, 0

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = criterion(model(bx), by)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

        val_probs, val_labels = evaluate_validation_subjects(model, X_val_list)
        current_j, current_th, current_acc, current_f1 = select_threshold_by_youden(val_probs, val_labels)

        if current_j > best_val_j:
            best_val_j = current_j
            best_thresh = current_th
            best_val_acc = current_acc
            best_val_f1 = current_f1
            best_epoch = epoch + 1
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_weights is None:
        return None

    model.load_state_dict(best_weights)
    model.eval()
    t_X_te = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            test_probs = torch.sigmoid(model(t_X_te)).detach().cpu().numpy()

    test_median_prob = float(np.median(test_probs))
    true_label = int(Y_te[0])
    test_pred = 1 if test_median_prob > best_thresh else 0
    test_correct = int(test_pred == true_label)
    signed_margin = (
        test_median_prob - best_thresh
        if true_label == 1
        else best_thresh - test_median_prob
    )

    result = {
        "true_label": true_label,
        "window_size_ms": window_size_ms,
        "stride_ms": stride_ms,
        "fold_seed": fold_seed,
        "train_windows": int(len(X_tr)),
        "test_windows": int(len(X_te)),
        "val_subjects": int(len(X_val_list)),
        "best_epoch": int(best_epoch),
        "val_j": float(best_val_j),
        "val_f1": float(best_val_f1),
        "val_acc": float(best_val_acc),
        "threshold": float(best_thresh),
        "test_median_prob": test_median_prob,
        "test_pred": int(test_pred),
        "test_correct": test_correct,
        "signed_margin": float(signed_margin),
    }

    del model, optimizer, scheduler, train_loader, t_X_tr, t_Y_tr, t_X_te
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def markdown_table(df, floatfmt=".4f"):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize_candidate_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()
    df["selection_rank_tuple"] = list(
        zip(df["val_j"], df["val_f1"], df["val_acc"], -df["window_size_ms"], -df["stride_ms"])
    )

    selected_rows = []
    fold_rows = []
    for fold, group in df.groupby("fold", sort=True):
        selected_idx = group["selection_rank_tuple"].idxmax()
        selected = group.loc[selected_idx].copy()
        selected_rows.append(selected)

        oracle_correct = int(group["test_correct"].max())
        oracle_windows = group.loc[group["test_correct"].eq(oracle_correct), "window_label"].tolist()
        fixed_1000 = group[(group["window_size_ms"].eq(1000)) & (group["stride_ms"].eq(500))]
        fixed_1000_correct = int(fixed_1000["test_correct"].iloc[0]) if len(fixed_1000) else np.nan
        fixed_1000_selected = bool(
            selected["window_size_ms"] == 1000 and selected["stride_ms"] == 500
        )

        fold_rows.append({
            "fold": int(fold),
            "pid": int(selected["pid"]),
            "true_label": int(selected["true_label"]),
            "selected_window": selected["window_label"],
            "selected_correct": int(selected["test_correct"]),
            "oracle_correct": oracle_correct,
            "selection_regret": int(oracle_correct - int(selected["test_correct"])),
            "oracle_correct_windows": ",".join(oracle_windows),
            "fixed_1000_500_correct": fixed_1000_correct,
            "fixed_1000_500_selected": fixed_1000_selected,
            "selected_val_j": float(selected["val_j"]),
            "selected_val_f1": float(selected["val_f1"]),
            "selected_val_acc": float(selected["val_acc"]),
            "selected_threshold": float(selected["threshold"]),
            "selected_test_prob": float(selected["test_median_prob"]),
        })

    selected_df = pd.DataFrame(selected_rows).drop(columns=["selection_rank_tuple"])
    fold_df = pd.DataFrame(fold_rows)
    selected_df.to_csv(os.path.join(output_dir, "selected_candidate_predictions.csv"), index=False)
    fold_df.to_csv(os.path.join(output_dir, "selected_vs_oracle_summary.csv"), index=False)

    window_summary = df.groupby("window_label").agg(
        n=("fold", "count"),
        test_accuracy=("test_correct", "mean"),
        mean_val_j=("val_j", "mean"),
        mean_val_f1=("val_f1", "mean"),
        mean_val_acc=("val_acc", "mean"),
        mean_signed_margin=("signed_margin", "mean"),
    ).reset_index()
    window_summary.to_csv(os.path.join(output_dir, "candidate_window_summary.csv"), index=False)

    selected_acc = selected_df["test_correct"].mean()
    selected_f1 = f1_score(selected_df["true_label"], selected_df["test_pred"])
    oracle_rate = fold_df["oracle_correct"].mean()
    regret_rate = fold_df["selection_regret"].mean()
    fixed = df[(df["window_size_ms"].eq(1000)) & (df["stride_ms"].eq(500))]
    fixed_acc = fixed["test_correct"].mean()
    fixed_f1 = f1_score(fixed["true_label"], fixed["test_pred"])

    core_summary = pd.DataFrame([
        {"metric": "selected_candidate_accuracy", "value": selected_acc},
        {"metric": "selected_candidate_f1", "value": selected_f1},
        {"metric": "oracle_candidate_correct_rate", "value": oracle_rate},
        {"metric": "selection_regret_rate", "value": regret_rate},
        {"metric": "fixed_1000_500_candidate_accuracy", "value": fixed_acc},
        {"metric": "fixed_1000_500_candidate_f1", "value": fixed_f1},
        {
            "metric": "folds_where_selected_wrong_but_other_candidate_correct",
            "value": int(fold_df["selection_regret"].sum()),
        },
        {
            "metric": "folds_where_1000_500_correct_but_not_selected",
            "value": int((fold_df["fixed_1000_500_correct"].eq(1) & ~fold_df["fixed_1000_500_selected"]).sum()),
        },
    ])
    core_summary.to_csv(os.path.join(output_dir, "validation_test_alignment_summary.csv"), index=False)

    md_path = os.path.join(output_dir, "validation_test_alignment.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CNNLSTM candidate-level nested selection diagnostic\n\n")
        f.write("## Core summary\n\n")
        f.write(markdown_table(core_summary))
        f.write("\n\n## Candidate window summary\n\n")
        f.write(markdown_table(window_summary))
        f.write("\n\n## Selected vs oracle summary\n\n")
        preview_cols = [
            "fold", "pid", "true_label", "selected_window", "selected_correct",
            "oracle_correct", "selection_regret", "oracle_correct_windows",
            "fixed_1000_500_correct", "fixed_1000_500_selected",
        ]
        f.write(markdown_table(fold_df[preview_cols]))
        f.write("\n")

    print("\nCore summary:")
    print(core_summary.to_string(index=False))
    print(f"\nSaved diagnostic report: {md_path}")


def run(data_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 40)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Output dir: {output_dir}")
    print("=" * 40)

    full_df = load_all_data(data_folder)
    check_mask_distribution(full_df)

    pid_to_label = {
        pid: 1 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0
        for pid, group in full_df.groupby(PID_COL)
    }
    all_pids = sorted(pid_to_label.keys())

    partial_path = os.path.join(output_dir, "candidate_level_results_partial.csv")
    if os.path.exists(partial_path):
        existing = pd.read_csv(partial_path)
        records = existing.to_dict("records")
        completed = set(zip(existing["fold"].astype(int), existing["window_label"].astype(str)))
        print(f"Resuming from partial results: {len(existing)} rows already completed.")
    else:
        records = []
        completed = set()

    for i, test_pid in enumerate(all_pids):
        fold = i + 1
        fold_seed = SEED + fold
        print(f"\n=== Fold {fold}/{len(all_pids)} | PID {test_pid} | fold_seed={fold_seed} ===")
        pool_pids = [p for p in all_pids if p != test_pid]
        pool_labels = [pid_to_label[p] for p in pool_pids]
        train_pids, val_pids = get_stratified_train_val_pids(
            pool_pids,
            pool_labels,
            test_size=0.15,
            random_state=SEED + i,
        )

        for window_size_ms, stride_ms in WINDOW_CANDIDATES:
            window_label = f"{window_size_ms}/{stride_ms}"
            if (fold, window_label) in completed:
                print(f"  Candidate {window_label} already completed, skipping.")
                continue

            print(f"  Candidate {window_label}")
            result = train_one_candidate(
                full_df,
                train_pids,
                val_pids,
                test_pid,
                window_size_ms,
                stride_ms,
                fold_seed,
            )
            if result is None:
                print("    skipped: no usable data/model")
                continue

            result.update({
                "fold": fold,
                "pid": int(test_pid),
                "window_label": window_label,
                "train_subjects": len(train_pids),
                "validation_subjects_available": len(val_pids),
            })
            records.append(result)
            print(
                f"    val_j={result['val_j']:.4f} val_f1={result['val_f1']:.4f} "
                f"th={result['threshold']:.2f} test_prob={result['test_median_prob']:.4f} "
                f"pred={result['test_pred']} correct={result['test_correct']}"
            )

        pd.DataFrame(records).to_csv(partial_path, index=False)

    df = pd.DataFrame(records)
    candidate_path = os.path.join(output_dir, "candidate_level_results.csv")
    df.to_csv(candidate_path, index=False)
    print(f"\nCandidate-level results saved: {candidate_path}")
    summarize_candidate_results(df, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CNNLSTM candidate-level nested selection diagnostic.")
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_folder, args.output_dir)
