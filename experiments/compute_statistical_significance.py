"""
Participant-level paired significance tests for model comparisons.

Primary test:
- Exact McNemar test on paired correctness indicators for the same held-out
  participants. This is appropriate for comparing two classifiers' accuracy
  on the identical LOSO test subjects.

Secondary analysis:
- Paired participant bootstrap for accuracy and F1 differences. This gives an
  effect-size confidence interval for metrics, especially F1, that are not
  directly tested by McNemar.

Run after `run_model_comparison.py` has generated:
  results/model_comparison/cnnlstm_subject_predictions.csv
and after `run_attentionnet_loso.py` has generated:
  results/attention/fold_level_metrics.csv
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from scipy.stats import binomtest
except ImportError:  # pragma: no cover - kept for older SciPy environments
    binomtest = None


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATTENTION_PATH = os.path.join(ROOT_DIR, "results", "attention", "fold_level_metrics.csv")
MODEL_COMPARISON_DIR = os.path.join(ROOT_DIR, "results", "model_comparison")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "statistical_tests")
BOOTSTRAP_ITERATIONS = 10000
SEED = 42


@dataclass
class PairedTestResult:
    comparison: str
    n_subjects: int
    model_a_accuracy: float
    model_b_accuracy: float
    accuracy_delta: float
    model_a_f1: float
    model_b_f1: float
    f1_delta: float
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_p: float
    bootstrap_acc_delta_low: float
    bootstrap_acc_delta_high: float
    bootstrap_acc_delta_p: float
    bootstrap_f1_delta_low: float
    bootstrap_f1_delta_high: float
    bootstrap_f1_delta_p: float


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant-pair counts."""
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    if binomtest is not None:
        return float(binomtest(min(b, c), n=n_discordant, p=0.5, alternative="two-sided").pvalue)

    # Fallback exact two-sided binomial probability.
    from math import comb

    tail = sum(comb(n_discordant, k) for k in range(0, min(b, c) + 1)) / (2 ** n_discordant)
    return float(min(1.0, 2 * tail))


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    acc_deltas = np.empty(BOOTSTRAP_ITERATIONS)
    f1_deltas = np.empty(BOOTSTRAP_ITERATIONS)

    for i in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        acc_deltas[i] = accuracy_score(yt, pa) - accuracy_score(yt, pb)
        f1_deltas[i] = f1_score(yt, pa, zero_division=0) - f1_score(yt, pb, zero_division=0)

    def two_sided_p(samples: np.ndarray) -> float:
        return float(min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0))))

    return {
        "acc_low": float(np.percentile(acc_deltas, 2.5)),
        "acc_high": float(np.percentile(acc_deltas, 97.5)),
        "acc_p": two_sided_p(acc_deltas),
        "f1_low": float(np.percentile(f1_deltas, 2.5)),
        "f1_high": float(np.percentile(f1_deltas, 97.5)),
        "f1_p": two_sided_p(f1_deltas),
    }


def load_attention_predictions() -> pd.DataFrame:
    if not os.path.exists(ATTENTION_PATH):
        raise FileNotFoundError(f"Missing AttentionNet predictions: {ATTENTION_PATH}")
    df = pd.read_csv(ATTENTION_PATH)
    return df.rename(columns={"median_prob": "score"})[
        ["pid", "true_label", "pred", "threshold", "score"]
    ].assign(model="AttentionNet")


def load_comparator_predictions(model_name: str) -> pd.DataFrame:
    path = os.path.join(MODEL_COMPARISON_DIR, f"{model_name.lower()}_subject_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {model_name} predictions: {path}\n"
            "Run experiments/run_model_comparison.py first to generate paired subject-level predictions."
        )
    df = pd.read_csv(path)
    return df.rename(columns={"median_prob": "score"})[
        ["pid", "true_label", "pred", "threshold", "score"]
    ].assign(model=model_name)


def compare_models(model_a: pd.DataFrame, model_b: pd.DataFrame) -> PairedTestResult:
    merged = model_a.merge(
        model_b,
        on="pid",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    ).sort_values("pid")

    if not (merged["true_label_a"].values == merged["true_label_b"].values).all():
        raise ValueError("Paired prediction files disagree on true labels for at least one PID.")

    y_true = merged["true_label_a"].to_numpy(dtype=int)
    pred_a = merged["pred_a"].to_numpy(dtype=int)
    pred_b = merged["pred_b"].to_numpy(dtype=int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # b: model A correct, model B wrong; c: model A wrong, model B correct.
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    boot = paired_bootstrap(y_true, pred_a, pred_b)

    return PairedTestResult(
        comparison=f"{model_a['model'].iloc[0]} vs {model_b['model'].iloc[0]}",
        n_subjects=len(merged),
        model_a_accuracy=float(accuracy_score(y_true, pred_a)),
        model_b_accuracy=float(accuracy_score(y_true, pred_b)),
        accuracy_delta=float(accuracy_score(y_true, pred_a) - accuracy_score(y_true, pred_b)),
        model_a_f1=float(f1_score(y_true, pred_a, zero_division=0)),
        model_b_f1=float(f1_score(y_true, pred_b, zero_division=0)),
        f1_delta=float(f1_score(y_true, pred_a, zero_division=0) - f1_score(y_true, pred_b, zero_division=0)),
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_p=exact_mcnemar_pvalue(b, c),
        bootstrap_acc_delta_low=boot["acc_low"],
        bootstrap_acc_delta_high=boot["acc_high"],
        bootstrap_acc_delta_p=boot["acc_p"],
        bootstrap_f1_delta_low=boot["f1_low"],
        bootstrap_f1_delta_high=boot["f1_high"],
        bootstrap_f1_delta_p=boot["f1_p"],
    )


def write_markdown(results: list[PairedTestResult], path: str) -> None:
    rows = []
    for r in results:
        rows.append(
            "| {comparison} | {n} | {acc_a:.3f} | {acc_b:.3f} | {acc_delta:.3f} | "
            "{b} | {c} | {mcnemar_p:.4f} | [{acc_lo:.3f}, {acc_hi:.3f}] | {acc_p:.4f} | "
            "{f1_delta:.3f} | [{f1_lo:.3f}, {f1_hi:.3f}] | {f1_p:.4f} |".format(
                comparison=r.comparison,
                n=r.n_subjects,
                acc_a=r.model_a_accuracy,
                acc_b=r.model_b_accuracy,
                acc_delta=r.accuracy_delta,
                b=r.mcnemar_b,
                c=r.mcnemar_c,
                mcnemar_p=r.mcnemar_p,
                acc_lo=r.bootstrap_acc_delta_low,
                acc_hi=r.bootstrap_acc_delta_high,
                acc_p=r.bootstrap_acc_delta_p,
                f1_delta=r.f1_delta,
                f1_lo=r.bootstrap_f1_delta_low,
                f1_hi=r.bootstrap_f1_delta_high,
                f1_p=r.bootstrap_f1_delta_p,
            )
        )

    content = "\n".join(
        [
            "# Paired Statistical Tests",
            "",
            "Exact McNemar is the primary test for paired participant-level correctness. "
            "Paired bootstrap reports effect-size intervals for metric differences.",
            "",
            "| Comparison | n | Acc A | Acc B | Delta Acc | b | c | McNemar p | Bootstrap Delta Acc 95% CI | Bootstrap p | Delta F1 | Bootstrap Delta F1 95% CI | Bootstrap p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Here b is the number of subjects correctly classified by model A and incorrectly classified by model B; "
            "c is the opposite discordant-pair count.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    attention = load_attention_predictions()
    results = []

    for comparator in ("CNNLSTM", "PureLSTM"):
        try:
            results.append(compare_models(attention, load_comparator_predictions(comparator)))
        except FileNotFoundError as exc:
            print(exc)

    if not results:
        raise SystemExit("No paired comparator predictions were available.")

    out_csv = os.path.join(OUTPUT_DIR, "paired_significance_tests.csv")
    out_md = os.path.join(OUTPUT_DIR, "paired_significance_tests.md")
    pd.DataFrame([r.__dict__ for r in results]).to_csv(out_csv, index=False)
    write_markdown(results, out_md)

    print(f"Saved paired significance CSV: {out_csv}")
    print(f"Saved paired significance summary: {out_md}")
    for r in results:
        print(
            f"{r.comparison}: McNemar b={r.mcnemar_b}, c={r.mcnemar_c}, "
            f"p={r.mcnemar_p:.4f}; delta acc={r.accuracy_delta:.3f}; delta F1={r.f1_delta:.3f}"
        )


if __name__ == "__main__":
    main()
