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
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
