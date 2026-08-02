"""Fast structural, archived-result, and path-hygiene checks for release v4."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FIXED = ROOT / "results" / "fixed_multiseed_strict"
NESTED = ROOT / "results" / "nested_multiseed_strict"


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(relative)
    return path


fixed = pd.read_csv(require("results/fixed_multiseed_strict/aggregate/multiseed_task_metrics.csv"))
nested = pd.read_csv(require("results/nested_multiseed_strict/aggregate/nested_task_metrics.csv"))
fixed_pair = pd.read_csv(require("results/fixed_multiseed_strict/aggregate/multiseed_paired_differences.csv"))
nested_pair = pd.read_csv(require("results/nested_multiseed_strict/aggregate/nested_paired_comparisons.csv"))
windows = pd.read_csv(require("results/nested_multiseed_strict/aggregate/nested_selected_window_by_fold.csv"))
sampling = pd.read_csv(require("results/subject_balancing_ablation_nested/subject_balancing_ablation_nested_summary.csv"))

assert len(fixed) == 75
assert len(nested) == 20
assert sorted(fixed["global_seed"].unique().tolist()) == [42, 43, 44, 45, 46]
assert sorted(nested["global_seed"].unique().tolist()) == [42, 43, 44, 45, 46]


def mean_accuracy(frame: pd.DataFrame, model: str, config: str) -> float:
    rows = frame[(frame["model"] == model) & (frame["config"] == config)]
    assert len(rows) == 5
    return float(rows["accuracy"].mean())


assert abs(mean_accuracy(fixed, "AttentionNet", "Full_Preprocessing") - 0.7649122807) < 1e-9
assert abs(mean_accuracy(fixed, "CNNLSTM", "Full_Preprocessing") - 0.7578947368) < 1e-9
assert abs(mean_accuracy(fixed, "PureLSTM", "Full_Preprocessing") - 0.7438596491) < 1e-9
assert abs(mean_accuracy(nested, "AttentionNet", "Full_Preprocessing") - 0.7684210526) < 1e-9
assert abs(mean_accuracy(nested, "CNNLSTM", "Full_Preprocessing") - 0.7578947368) < 1e-9
assert abs(mean_accuracy(nested, "PureLSTM", "Full_Preprocessing") - 0.7508771930) < 1e-9
assert abs(mean_accuracy(nested, "AttentionNet", "No_Filtering") - 0.7789473684) < 1e-9

ci_columns = [column for column in fixed_pair if column.endswith("_ci_low")]
assert ci_columns
for low_column in ci_columns:
    high_column = low_column.replace("_ci_low", "_ci_high")
    if high_column in fixed_pair:
        assert ((fixed_pair[low_column] <= 0) & (fixed_pair[high_column] >= 0)).all()

assert len(windows) == 1140
assert sampling["mode"].tolist() == ["balanced", "unbalanced"]
assert abs(float(sampling.loc[sampling["mode"] == "balanced", "accuracy"].iloc[0]) - 0.7719298246) < 1e-9
assert len(list(FIXED.glob("*/*/seed_*/fold_predictions.csv"))) == 75
assert len(list(NESTED.glob("*/*/seed_*/fold_predictions.csv"))) == 20
assert len(list(FIXED.glob("*/*/seed_*/folds/fold_*.json"))) == 4275
assert len(list(NESTED.glob("*/*/seed_*/folds/fold_*.json"))) == 1140

required_scripts = [
    "experiments/run_fixed_multiseed_fold_strict.py",
    "experiments/run_nested_multiseed_fold_strict.py",
    "experiments/aggregate_fixed_multiseed_strict.py",
    "experiments/aggregate_nested_multiseed_strict.py",
    "figures/generate_protocol_sensitivity_figure.py",
]
for script in required_scripts:
    require(script)

# Historical task contracts intentionally retain execution-host paths as provenance.
for directory in (ROOT / "experiments", ROOT / "figures", ROOT / "data_processing"):
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        forbidden = ["D:" + "\\development\\", "/root/" + "autodl-tmp"]
        if any(token in text for token in forbidden):
            raise AssertionError(f"Nonportable path found in {path.relative_to(ROOT)}")

assert not nested_pair.empty
print("Release v4 smoke check passed: structure, strict metrics, fold counts, and path hygiene are consistent.")
