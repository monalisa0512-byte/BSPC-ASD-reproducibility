"""
Analyze why AttentionNet under nested window selection underperforms CNNLSTM.

This script does not train models. It combines existing fold-level predictions
with per-subject data-quality and valid-window diagnostics.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from run_attentionnet_loso import (
    DATA_FOLDER,
    MASK_FEATURES,
    PID_COL,
    TRIAL_COL,
    WINDOW_CANDIDATES,
    count_valid_windows_for_pids,
    load_all_data,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "nested_reversal_analysis"


def read_predictions(path):
    df = pd.read_csv(path)
    if "predicted_label" in df.columns and "pred" not in df.columns:
        df = df.rename(columns={"predicted_label": "pred"})
    df["true_label"] = df["true_label"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["correct"] = (df["true_label"] == df["pred"]).astype(int)
    if "median_prob" in df.columns and "threshold" in df.columns:
        df["signed_margin"] = np.where(
            df["true_label"].eq(1),
            df["median_prob"] - df["threshold"],
            df["threshold"] - df["median_prob"],
        )
        df["abs_margin"] = df["signed_margin"].abs()
    return df


def metric_row(name, df):
    y = df["true_label"].astype(int)
    p = df["pred"].astype(int)
    cm = confusion_matrix(y, p, labels=[0, 1])
    return {
        "model_setting": name,
        "n": len(df),
        "accuracy": accuracy_score(y, p),
        "f1": f1_score(y, p),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "wrong_pids": ",".join(str(x) for x in df.loc[df["correct"].eq(0), "pid"].tolist()),
    }


def subject_quality(full_df):
    rows = []
    valid_counts_by_candidate = {
        candidate: count_valid_windows_for_pids(full_df, full_df[PID_COL].dropna().unique(), *candidate)
        for candidate in WINDOW_CANDIDATES
    }

    for pid, group in full_df.groupby(PID_COL):
        mask = group[MASK_FEATURES].astype(float)
        frame_missing = mask.mean(axis=1)
        severe_missing = frame_missing.ge(0.5)
        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()
        row = {
            "pid": int(pid),
            "n_rows": int(len(group)),
            "n_trials": int(block_ids.nunique()),
            "mask_missing_rate": float(mask.values.mean()),
            "severe_frame_rate": float(severe_missing.mean()),
        }
        for window_size, stride in WINDOW_CANDIDATES:
            row[f"valid_windows_{window_size}_{stride}"] = int(
                valid_counts_by_candidate[(window_size, stride)].get(pid, 0)
            )
        rows.append(row)

    return pd.DataFrame(rows)


def add_window_count(df):
    col = "valid_windows_" + df["window_size_ms"].astype(int).astype(str) + "_" + df["stride_ms"].astype(int).astype(str)
    df = df.copy()
    df["selected_valid_windows"] = [df.loc[idx, c] for idx, c in zip(df.index, col)]
    return df


def safe_spearman(x, y):
    if len(pd.Series(x).dropna().unique()) < 2 or len(pd.Series(y).dropna().unique()) < 2:
        return np.nan, np.nan
    r, p = spearmanr(x, y, nan_policy="omit")
    return float(r), float(p)


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
                if col == "p" and np.isfinite(value) and value < 0.0001:
                    cells.append("<0.0001")
                else:
                    cells.append(format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "att_nested": RESULTS / "attention_nested_window" / "fold_level_metrics.csv",
        "cnn_nested": RESULTS / "model_comparison_nested_window" / "cnnlstm_subject_predictions.csv",
        "att_fixed": RESULTS / "attention" / "fold_level_metrics.csv",
        "cnn_fixed": RESULTS / "model_comparison" / "cnnlstm_subject_predictions.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))

    preds = {name: read_predictions(path) for name, path in paths.items()}
    metrics = pd.DataFrame(metric_row(name, df) for name, df in preds.items())
    metrics.to_csv(OUT_DIR / "model_level_metrics.csv", index=False)

    print("Loading processed data for quality diagnostics...")
    full_df = load_all_data(DATA_FOLDER)
    quality = subject_quality(full_df)
    quality.to_csv(OUT_DIR / "subject_quality_window_counts.csv", index=False)

    att = preds["att_nested"].merge(quality, on="pid", how="left")
    cnn = preds["cnn_nested"].merge(quality, on="pid", how="left")
    att = add_window_count(att)
    cnn = add_window_count(cnn)

    paired = att.merge(
        cnn,
        on="pid",
        suffixes=("_att", "_cnn"),
        validate="one_to_one",
    )
    paired["same_window"] = (
        paired["window_size_ms_att"].astype(int).eq(paired["window_size_ms_cnn"].astype(int))
        & paired["stride_ms_att"].astype(int).eq(paired["stride_ms_cnn"].astype(int))
    )
    paired["outcome_group"] = np.select(
        [
            paired["correct_att"].eq(1) & paired["correct_cnn"].eq(1),
            paired["correct_att"].eq(0) & paired["correct_cnn"].eq(0),
            paired["correct_att"].eq(1) & paired["correct_cnn"].eq(0),
            paired["correct_att"].eq(0) & paired["correct_cnn"].eq(1),
        ],
        ["both_correct", "both_wrong", "attention_only_correct", "cnnlstm_only_correct"],
        default="unknown",
    )
    paired["att_minus_cnn_margin"] = paired["signed_margin_att"] - paired["signed_margin_cnn"]
    paired.to_csv(OUT_DIR / "paired_pid_level_analysis.csv", index=False)

    same_rows = []
    for label, subset in [("same_selected_window", paired[paired["same_window"]]), ("different_selected_window", paired[~paired["same_window"]])]:
        if len(subset) == 0:
            continue
        same_rows.append({
            "subset": label,
            "n": len(subset),
            "attention_acc": accuracy_score(subset["true_label_att"], subset["pred_att"]),
            "cnnlstm_acc": accuracy_score(subset["true_label_cnn"], subset["pred_cnn"]),
            "attention_f1": f1_score(subset["true_label_att"], subset["pred_att"]),
            "cnnlstm_f1": f1_score(subset["true_label_cnn"], subset["pred_cnn"]),
        })
    same_window_summary = pd.DataFrame(same_rows)
    same_window_summary.to_csv(OUT_DIR / "same_vs_different_window_summary.csv", index=False)

    group_summary = paired.groupby("outcome_group").agg(
        n=("pid", "count"),
        mean_att_margin=("signed_margin_att", "mean"),
        mean_cnn_margin=("signed_margin_cnn", "mean"),
        mean_att_abs_margin=("abs_margin_att", "mean"),
        mean_cnn_abs_margin=("abs_margin_cnn", "mean"),
        mean_missing_rate=("mask_missing_rate_att", "mean"),
        mean_severe_frame_rate=("severe_frame_rate_att", "mean"),
        mean_selected_windows_att=("selected_valid_windows_att", "mean"),
        mean_selected_windows_cnn=("selected_valid_windows_cnn", "mean"),
    ).reset_index()
    group_summary.to_csv(OUT_DIR / "outcome_group_summary.csv", index=False)

    fixed_nested_rows = []
    for model in ["att", "cnn"]:
        fixed = preds[f"{model}_fixed"].set_index("pid")
        nested = preds[f"{model}_nested"].set_index("pid")
        common = fixed.index.intersection(nested.index)
        changed = pd.DataFrame({
            "pid": common,
            "true_label": fixed.loc[common, "true_label"].astype(int).values,
            "fixed_pred": fixed.loc[common, "pred"].astype(int).values,
            "nested_pred": nested.loc[common, "pred"].astype(int).values,
            "fixed_correct": fixed.loc[common, "correct"].astype(int).values,
            "nested_correct": nested.loc[common, "correct"].astype(int).values,
        })
        changed["model"] = "AttentionNet" if model == "att" else "CNNLSTM"
        fixed_nested_rows.append(changed)
    fixed_nested = pd.concat(fixed_nested_rows, ignore_index=True)
    fixed_nested.to_csv(OUT_DIR / "fixed_to_nested_pid_changes.csv", index=False)

    corr_rows = []
    for name, df in [("AttentionNet_nested", att), ("CNNLSTM_nested", cnn)]:
        for x in ["mask_missing_rate", "severe_frame_rate", "selected_valid_windows", "abs_margin", "signed_margin"]:
            r, p = safe_spearman(df[x], df["correct"])
            corr_rows.append({"model": name, "x": x, "y": "correct", "spearman_r": r, "p": p})
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(OUT_DIR / "quality_margin_correlations.csv", index=False)

    discordant = paired[paired["correct_att"].ne(paired["correct_cnn"])][[
        "pid", "true_label_att", "pred_att", "pred_cnn",
        "median_prob_att", "threshold_att", "signed_margin_att",
        "median_prob_cnn", "threshold_cnn", "signed_margin_cnn",
        "window_size_ms_att", "stride_ms_att", "window_size_ms_cnn", "stride_ms_cnn",
        "mask_missing_rate_att", "severe_frame_rate_att",
        "selected_valid_windows_att", "selected_valid_windows_cnn",
        "outcome_group",
    ]]
    discordant.to_csv(OUT_DIR / "discordant_subjects.csv", index=False)

    md_path = OUT_DIR / "nested_reversal_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Nested-window reversal analysis\n\n")
        f.write("## Model-level metrics\n\n")
        f.write(markdown_table(metrics.drop(columns=["wrong_pids"])))
        f.write("\n\n")
        f.write("## Same vs different selected windows\n\n")
        f.write(markdown_table(same_window_summary))
        f.write("\n\n")
        f.write("## Outcome groups\n\n")
        f.write(markdown_table(group_summary))
        f.write("\n\n")
        f.write("## Quality/margin correlations with correctness\n\n")
        f.write(markdown_table(correlations))
        f.write("\n\n")
        f.write("## Discordant subjects\n\n")
        f.write(markdown_table(discordant))
        f.write("\n")

    print(f"Analysis written to: {OUT_DIR}")
    print(metrics[["model_setting", "accuracy", "f1", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print("\nSame vs different selected windows:")
    print(same_window_summary.to_string(index=False))
    print("\nOutcome groups:")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
