from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "preprocessing_ablation_paired_tests"

MODEL_DIRS = {
    "AttentionNet": RESULTS / "attentionnet_preprocessing_ablation_pipeline_fixed",
    "CNNLSTM": RESULTS / "cnnlstm_preprocessing_ablation_pipeline_fixed",
    "PureLSTM": RESULTS / "purelstm_preprocessing_ablation_pipeline_fixed",
}

BASELINE = "Full_Preprocessing"
ABLATIONS = [
    "Linear_Interpolation",
    "Without_Blink_Expansion",
    "No_Filtering",
    "Without_Mask_Features",
]


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) * (0.5**n) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail)


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(20260707)
    n = len(y_true)
    acc_delta = []
    f1_delta = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        acc_delta.append(accuracy_score(yt, pa) - accuracy_score(yt, pb))
        f1_delta.append(f1_score(yt, pa, zero_division=0) - f1_score(yt, pb, zero_division=0))
    acc_delta = np.asarray(acc_delta)
    f1_delta = np.asarray(f1_delta)
    def two_sided_empirical_p(samples: np.ndarray) -> float:
        # With B=5,000 resamples, a zero observed tail is reported at the
        # finite-resolution bound 2/B rather than as p=0.
        tail = min(np.mean(samples <= 0), np.mean(samples >= 0))
        return float(min(1.0, 2 * max(tail, 1 / n_boot)))

    return {
        "delta_acc_ci_low": float(np.percentile(acc_delta, 2.5)),
        "delta_acc_ci_high": float(np.percentile(acc_delta, 97.5)),
        "bootstrap_acc_p": two_sided_empirical_p(acc_delta),
        "delta_f1_ci_low": float(np.percentile(f1_delta, 2.5)),
        "delta_f1_ci_high": float(np.percentile(f1_delta, 97.5)),
        "bootstrap_f1_p": two_sided_empirical_p(f1_delta),
    }


def load_predictions(model_dir: Path, config: str) -> pd.DataFrame:
    path = model_dir / f"subject_predictions_{config}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = {"pid", "true_label", "pred_label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["pid", "true_label", "pred_label"]].copy()


def compare(model: str, model_dir: Path, ablation: str) -> dict[str, object]:
    full = load_predictions(model_dir, BASELINE).rename(columns={"pred_label": "pred_full"})
    abl = load_predictions(model_dir, ablation).rename(columns={"pred_label": "pred_ablation"})
    merged = full.merge(abl, on=["pid", "true_label"], how="inner")
    if len(merged) != len(full) or len(merged) != len(abl):
        raise ValueError(f"{model} {ablation}: PID mismatch after merge")

    y_true = merged["true_label"].to_numpy()
    pred_full = merged["pred_full"].to_numpy()
    pred_ablation = merged["pred_ablation"].to_numpy()
    full_correct = pred_full == y_true
    ablation_correct = pred_ablation == y_true

    b = int(np.sum(full_correct & ~ablation_correct))
    c = int(np.sum(~full_correct & ablation_correct))
    boot = paired_bootstrap(y_true, pred_full, pred_ablation)

    return {
        "model": model,
        "comparison": f"{BASELINE} vs {ablation}",
        "n": int(len(merged)),
        "full_acc": float(accuracy_score(y_true, pred_full)),
        "ablation_acc": float(accuracy_score(y_true, pred_ablation)),
        "delta_acc": float(accuracy_score(y_true, pred_full) - accuracy_score(y_true, pred_ablation)),
        "full_f1": float(f1_score(y_true, pred_full, zero_division=0)),
        "ablation_f1": float(f1_score(y_true, pred_ablation, zero_division=0)),
        "delta_f1": float(f1_score(y_true, pred_full, zero_division=0) - f1_score(y_true, pred_ablation, zero_division=0)),
        "b_full_correct_ablation_wrong": b,
        "c_full_wrong_ablation_correct": c,
        "mcnemar_p": exact_mcnemar_p(b, c),
        **boot,
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_p(x: float) -> str:
    return "<0.0004" if x <= 0.0004 else f"{x:.4f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, model_dir in MODEL_DIRS.items():
        for ablation in ABLATIONS:
            rows.append(compare(model, model_dir, ablation))

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "preprocessing_ablation_paired_tests.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md = [
        "# Preprocessing Ablation Paired Statistical Tests",
        "",
        "Exact McNemar is computed on paired participant-level correctness. Paired bootstrap reports Full-minus-ablation effect-size intervals for Accuracy and F1. With B=5,000 resamples, an empirical two-sided p value at the finite-resolution bound is shown as <0.0004 rather than p=0.",
        "",
        "| Model | Comparison | n | Full Acc | Ablation Acc | Delta Acc | Full F1 | Ablation F1 | Delta F1 | b | c | McNemar p | Delta Acc 95% CI | Bootstrap Acc p | Delta F1 95% CI | Bootstrap F1 p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        md.append(
            "| {model} | {comparison} | {n} | {full_acc} | {ablation_acc} | {delta_acc} | "
            "{full_f1} | {ablation_f1} | {delta_f1} | {b} | {c} | {mcnemar_p:.4f} | "
            "[{acc_lo}, {acc_hi}] | {boot_acc_p} | [{f1_lo}, {f1_hi}] | {boot_f1_p} |".format(
                model=row["model"],
                comparison=row["comparison"],
                n=int(row["n"]),
                full_acc=fmt_pct(row["full_acc"]),
                ablation_acc=fmt_pct(row["ablation_acc"]),
                delta_acc=fmt_pct(row["delta_acc"]),
                full_f1=fmt_pct(row["full_f1"]),
                ablation_f1=fmt_pct(row["ablation_f1"]),
                delta_f1=fmt_pct(row["delta_f1"]),
                b=int(row["b_full_correct_ablation_wrong"]),
                c=int(row["c_full_wrong_ablation_correct"]),
                mcnemar_p=row["mcnemar_p"],
                acc_lo=fmt_pct(row["delta_acc_ci_low"]),
                acc_hi=fmt_pct(row["delta_acc_ci_high"]),
                boot_acc_p=fmt_p(row["bootstrap_acc_p"]),
                f1_lo=fmt_pct(row["delta_f1_ci_low"]),
                f1_hi=fmt_pct(row["delta_f1_ci_high"]),
                boot_f1_p=fmt_p(row["bootstrap_f1_p"]),
            )
        )
    md.append("")
    md.append("Here b is the number of subjects correctly classified by Full preprocessing and incorrectly classified by the ablation; c is the opposite discordant-pair count.")
    md_path = OUT_DIR / "preprocessing_ablation_paired_tests.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    pri…81533 tokens truncated…

def make_train_loader(X_tr, Y_tr, P_tr, balanced):
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    t_Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(t_X_tr, t_Y_tr)

    if not balanced:
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), t_X_tr, t_Y_tr

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
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler), t_X_tr, t_Y_tr


def evaluate_subjects(model, X_val_list):
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for t_X_v, y_v in X_val_list:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                probs = torch.sigmoid(model(t_X_v)).detach().cpu().numpy()
            val_probs.append(float(np.median(probs)))
            val_labels.append(int(y_v))
    return val_probs, val_labels


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


def train_candidate(full_df, train_pids, val_pids, test_pid, window_size_ms, stride_ms, balanced, fold_idx):
    out = create_dataset_from_full_data_with_starts(
        full_df,
        train_pids,
        test_pid,
        window_size_ms=window_size_ms,
        stride_ms=stride_ms,
    )
    if out[0] is None or len(out[0]) == 0:
        return None

    X_tr, Y_tr, P_tr, X_te, Y_te, starts_te, scaler = out

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

    train_loader, t_X_tr, t_Y_tr = make_train_loader(X_tr, Y_tr, P_tr, balanced=balanced)
    model = AttentionNet(input_dim=X_tr.shape[2]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_j, best_val_f1, best_val_acc = -1.0, 0.0, 0.0
    best_thresh, best_model_weights = 0.5, None
    patience, patience_counter = 12, 0

    desc = f"Fold {fold_idx} {window_size_ms}/{stride_ms} {'balanced' if balanced else 'unbalanced'}"
    for _ in tqdm(range(EPOCHS), desc=desc, leave=False):
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

        probs, labels = evaluate_subjects(model, X_val_list)
        current_j, current_th, current_acc, current_f1 = select_threshold_by_youden(probs, labels)

        if current_j > best_val_j:
            best_val_j = current_j
            best_thresh = current_th
            best_val_acc = current_acc
            best_val_f1 = current_f1
            best_model_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    result = {
        "window_size_ms": window_size_ms,
        "stride_ms": stride_ms,
        "weights": best_model_weights,
        "input_dim": X_tr.shape[2],
        "X_te": X_te,
        "Y_te": Y_te,
        "starts_te": starts_te,
        "threshold": best_thresh,
        "val_j": best_val_j,
        "val_f1": best_val_f1,
        "val_acc": best_val_acc,
        "train_windows": len(X_tr),
        "val_subjects": len(X_val_list),
    }

    del model, optimizer, scheduler, train_loader, t_X_tr, t_Y_tr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_ablation_mode(full_df, output_dir, balanced):
    set_seeds(SEED)
    mode_name = "balanced" if balanced else "unbalanced"
    mode_dir = os.path.join(output_dir, mode_name)
    os.makedirs(mode_dir, exist_ok=True)

    pid_to_label = {
        pid: 1 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0
        for pid, group in full_df.groupby(PID_COL)
    }
    all_pids = sorted(pid_to_label.keys())

    records = []
    y_true_all, y_pred_all = [], []
    for i, test_pid in enumerate(all_pids):
        print(f"\n=== {mode_name} | Fold {i + 1}/{len(all_pids)} | Test PID: {test_pid} ===")
        pool_pids = [p for p in all_pids if p != test_pid]
        pool_labels = [pid_to_label[p] for p in pool_pids]
        train_pids, val_pids = get_stratified_train_val_pids(
            pool_pids, pool_labels, test_size=0.15, random_state=SEED + i
        )

        selected_candidate, selected_score = None, None
        for window_size_ms, stride_ms in WINDOW_CANDIDATES:
            print(f"  Candidate window -> {window_size_ms}/{stride_ms} ms")
            candidate = train_candidate(
                full_df,
                train_pids,
                val_pids,
                test_pid,
                window_size_ms,
                stride_ms,
                balanced=balanced,
                fold_idx=i + 1,
            )
            if candidate is None or candidate["weights"] is None:
                print("    Skipping candidate: no usable model.")
                continue

            score = (
                candidate["val_j"],
                candidate["val_f1"],
                candidate["val_acc"],
                -window_size_ms,
                -stride_ms,
            )
            print(
                f"    Val J={candidate['val_j']:.4f} | "
                f"F1={candidate['val_f1']:.4f} | Acc={candidate['val_acc']:.4f} | "
                f"Threshold={candidate['threshold']:.2f}"
            )
            if selected_score is None or score > selected_score:
                selected_score = score
                selected_candidate = candidate

        if selected_candidate is None:
            print(f"  Fold {i + 1} skipped: no valid nested candidate.")
            continue

        model = AttentionNet(input_dim=selected_candidate["input_dim"]).to(DEVICE)
        model.load_state_dict(selected_candidate["weights"])
        model.eval()
        t_X_te = torch.tensor(selected_candidate["X_te"], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                test_probs = torch.sigmoid(model(t_X_te)).detach().cpu().numpy()

        median_prob = float(np.median(test_probs))
        true_label = int(selected_candidate["Y_te"][0])
        pred = 1 if median_prob > selected_candidate["threshold"] else 0
        y_true_all.append(true_label)
        y_pred_all.append(pred)

        records.append({
            "mode": mode_name,
            "fold": i + 1,
            "pid": test_pid,
            "true_label": true_label,
            "pred": pred,
            "correct": int(pred == true_label),
            "window_size_ms": selected_candidate["window_size_ms"],
            "stride_ms": selected_candidate["stride_ms"],
            "inner_val_j": selected_candidate["val_j"],
            "inner_val_f1": selected_candidate["val_f1"],
            "inner_val_acc": selected_candidate["val_acc"],
            "threshold": selected_candidate["threshold"],
            "median_prob": median_prob,
            "train_windows": selected_candidate["train_windows"],
            "val_subjects": selected_candidate["val_subjects"],
        })

        print(
            f"  Selected {selected_candidate['window_size_ms']}/{selected_candidate['stride_ms']} | "
            f"True={true_label}, Median prob={median_prob:.4f}, Pred={pred}"
        )

        del model, t_X_te
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    csv_path = os.path.join(mode_dir, "fold_level_metrics.csv")
    df.to_csv(csv_path, index=False)

    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    print(f"\n{mode_name} final Accuracy={acc * 100:.2f}% | F1={f1 * 100:.2f}%")
    print(cm)

    return df


def load_existing_balanced_if_available(output_dir):
    local_path = os.path.join(output_dir, "balanced", "fold_level_metrics.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df["mode"] = "balanced"
        return df

    previous_path = os.path.join(ATTENTION_OUTPUT_DIR + "_nested_window", "fold_level_metrics.csv")
    if os.path.exists(previous_path):
        df = pd.read_csv(previous_path)
        df = df.rename(columns={"accuracy": "correct"})
        df["mode"] = "balanced"
        keep_cols = [
            "mode", "fold", "pid", "true_label", "pred", "correct", "window_size_ms", "stride_ms",
            "inner_val_j", "inner_val_f1", "inner_val_acc", "threshold", "median_prob",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
        os.makedirs(os.path.join(output_dir, "balanced"), exist_ok=True)
        df.to_csv(local_path, index=False)
        return df
    return None


def paired_bootstrap_delta(df_a, df_b, metric, n_boot=5000, seed=SEED):
    merged = df_a.merge(
        df_b,
        on="pid",
        suffixes=("_balanced", "_unbalanced"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    deltas = []
    pids = merged["pid"].to_numpy()
    for _ in range(n_boot):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        sample = merged.set_index("pid").loc[sampled].reset_index()
        if metric == "accuracy":
            m_bal = accuracy_score(sample["true_label_balanced"], sample["pred_balanced"])
            m_unbal = accuracy_score(sample["true_label_unbalanced"], sample["pred_unbalanced"])
        elif metric == "f1":
            m_bal = f1_score(sample["true_label_balanced"], sample["pred_balanced"])
            m_unbal = f1_score(sample["true_label_unbalanced"], sample["pred_unbalanced"])
        else:
            raise ValueError(metric)
        deltas.append(m_bal - m_unbal)
    deltas = np.array(deltas)
    observed = (
        accuracy_score(merged["true_label_balanced"], merged["pred_balanced"])
        - accuracy_score(merged["true_label_unbalanced"], merged["pred_unbalanced"])
        if metric == "accuracy"
        else f1_score(merged["true_label_balanced"], merged["pred_balanced"])
        - f1_score(merged["true_label_unbalanced"], merged["pred_unbalanced"])
    )
    p_value = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return observed, np.percentile(deltas, [2.5, 97.5]), min(1.0, float(p_value))


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


def write_summary(output_dir, df_balanced, df_unbalanced):
    os.makedirs(output_dir, exist_ok=True)
    combined = pd.concat([df_balanced, df_unbalanced], ignore_index=True)
    combined_path = os.path.join(output_dir, "subject_balancing_ablation_nested_combined.csv")
    combined.to_csv(combined_path, index=False)

    rows = []
    for mode, df in combined.groupby("mode"):
        y_true = df["true_label"].astype(int)
        y_pred = df["pred"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        rows.append({
            "mode": mode,
            "n_subjects": len(df),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })

    summary = pd.DataFrame(rows).sort_values("mode")
    summary_path = os.path.join(output_dir, "subject_balancing_ablation_nested_summary.csv")
    summary.to_csv(summary_path, index=False)

    merged = df_balanced.merge(
        df_unbalanced,
        on="pid",
        suffixes=("_balanced", "_unbalanced"),
        validate="one_to_one",
    )
    bal_correct = merged["pred_balanced"].astype(int) == merged["true_label_balanced"].astype(int)
    unbal_correct = merged["pred_unbalanced"].astype(int) == merged["true_label_unbalanced"].astype(int)
    b = int((bal_correct & ~unbal_correct).sum())
    c = int((~bal_correct & unbal_correct).sum())
    mcnemar_p = 1.0 if b + c == 0 else binomtest(min(b, c), n=b + c, p=0.5).pvalue
    delta_acc, ci_acc, p_acc = paired_bootstrap_delta(df_balanced, df_unbalanced, "accuracy")
    delta_f1, ci_f1, p_f1 = paired_bootstrap_delta(df_balanced, df_unbalanced, "f1")

    md_path = os.path.join(output_dir, "subject_balancing_ablation_nested.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Subject-balanced sampling ablation with nested window selection\n\n")
        f.write(markdown_table(summary))
        f.write("\n\n")
        f.write("## Paired comparison: balanced - unbalanced\n\n")
        f.write(f"- McNemar discordant counts: b={b}, c={c}, p={mcnemar_p:.4f}\n")
        f.write(
            f"- Accuracy delta: {delta_acc:.4f}, 95% bootstrap CI "
            f"[{ci_acc[0]:.4f}, {ci_acc[1]:.4f}], p={p_acc:.4f}\n"
        )
        f.write(
            f"- F1 delta: {delta_f1:.4f}, 95% bootstrap CI "
            f"[{ci_f1[0]:.4f}, {ci_f1[1]:.4f}], p={p_f1:.4f}\n"
        )
        f.write("\n## Window selections\n\n")
        for mode, df in combined.groupby("mode"):
            f.write(f"### {mode}\n\n")
            counts = (
                df.groupby(["window_size_ms", "stride_ms"])
                .size()
                .reset_index(name="count")
                .sort_values(["window_size_ms", "stride_ms"])
            )
            f.write(markdown_table(counts))
            f.write("\n\n")

    print(f"\nCombined results saved: {combined_path}")
    print(f"Summary saved: {summary_path}")
    print(f"Markdown report saved: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run nested-window subject-balanced sampling ablation.")
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["balanced", "unbalanced"],
        default=["unbalanced"],
        help="Modes to run. Existing balanced nested AttentionNet output is reused unless balanced is requested.",
    )
    parser.add_argument(
        "--no-reuse-balanced",
        action="store_true",
        help="Do not reuse previous balanced nested-window AttentionNet results.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only summarize existing balanced and unbalanced fold_level_metrics.csv files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 40)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 40)

    if args.summary_only:
        balanced_path = os.path.join(args.output_dir, "balanced", "fold_level_metrics.csv")
        unbalanced_path = os.path.join(args.output_dir, "unbalanced", "fold_level_metrics.csv")
        if not os.path.exists(balanced_path) or not os.path.exists(unbalanced_path):
            raise FileNotFoundError("summary-only requires existing balanced and unbalanced fold metrics.")
        df_balanced = pd.read_csv(balanced_path)
        df_unbalanced = pd.read_csv(unbalanced_path)
        df_balanced["mode"] = "balanced"
        df_unbalanced["mode"] = "unbalanced"
        write_summary(args.output_dir, df_balanced, df_unbalanced)
        return

    full_df = load_all_data(args.data_folder)
    check_mask_distribution(full_df)

    df_balanced = None
    if "balanced" in args.modes:
        df_balanced = run_ablation_mode(full_df, args.output_dir, balanced=True)
    elif not args.no_reuse_balanced:
        df_balanced = load_existing_balanced_if_available(args.output_dir)
        if df_balanced is not None:
            print("Reused existing balanced nested-window AttentionNet results.")

    df_unbalanced = None
    if "unbalanced" in args.modes:
        df_unbalanced = run_ablation_mode(full_df, args.output_dir, balanced=False)
    else:
        unbalanced_path = os.path.join(args.output_dir, "unbalanced", "fold_level_metrics.csv")
        if os.path.exists(unbalanced_path):
            df_unbalanced = pd.read_csv(unbalanced_path)
            df_unbalanced["mode"] = "unbalanced"

    if df_balanced is not None and df_unbalanced is not None:
        write_summary(args.output_dir, df_balanced, df_unbalanced)
    else:
        print("Summary not written because one mode is missing.")


if __name__ == "__main__":
    main()
