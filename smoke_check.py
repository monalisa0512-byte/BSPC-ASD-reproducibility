"""Fast structural and archived-result checks for the public release."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent


def require(relative):
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(relative)
    return path


fixed = pd.read_csv(require("results/attention/fold_level_metrics.csv"))
assert len(fixed) == 57
assert fixed["pred"].eq(fixed["true_label"]).sum() == 49

seeds = pd.read_csv(require("results/seed_sensitivity_fixed_1000_500_server_v2/seed_summary_all.csv"))
assert sorted(seeds["seed"].tolist()) == [42, 43, 44, 45, 46]
assert abs(seeds["accuracy"].mean() - 0.7859649123) < 1e-9

nested_att = pd.read_csv(require("results/attention_nested_window/fold_level_metrics.csv"))
nested_cnn = pd.read_csv(require("results/model_comparison_nested_window/cnnlstm_subject_predictions.csv"))
assert nested_att["pred"].eq(nested_att["true_label"]).sum() == 44
assert nested_cnn["pred"].eq(nested_cnn["true_label"]).sum() == 47

for path in ROOT.rglob("*"):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    if path.name == "smoke_check.py":
        continue
    if path.suffix.lower() not in {".py", ".sh", ".md", ".csv", ".json", ".txt", ".yml", ".yaml", ".cff"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    forbidden = [
        ":\\" + "Users\\",
        "D:\\" + "development\\",
        "/root/" + "autodl",
        "/" + "home/",
    ]
    if any(token in text for token in forbidden):
        raise AssertionError(f"Private or absolute path found in {path.relative_to(ROOT)}")

print("Release smoke check passed: required files, archived metrics, and path hygiene are consistent.")
