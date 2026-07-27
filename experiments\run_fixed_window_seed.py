import argparse
import os
import random
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import run_attentionnet_loso as attention_exp
import run_model_comparison as comparison_exp


DEFAULT_WINDOW_SIZE = 1000
DEFAULT_STRIDE = 500


def set_experiment_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_module(module, seed: int, window_size: int, stride: int) -> None:
    module.SEED = seed
    module.WINDOW_SIZE = window_size
    module.STRIDE = stride
    module.WINDOW_CANDIDATES = [(window_size, stride)]


def disable_attention_plots() -> None:
    def _skip_plot(*args, **kwargs):
        return None

    attention_exp.plot_attention_for_subject = _skip_plot


def summarize_predictions(csv_path: str) -> dict[str, Union[float, int, str]]:
    df = pd.read_csv(csv_path)
    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()
    return {
        "n_subjects": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "errors": ",".join(str(pid) for pid in df.loc[y_true != y_pred, "pid"].tolist()),
    }


def run(
    seed: int,
    data_folder: str,
    output_root: str,
    window_size: int,
    stride: int,
    models: list[str],
) -> None:
    set_experiment_seed(seed)
    configure_module(attention_exp, seed, window_size, stride)
    configure_module(comparison_exp, seed, window_size, stride)
    disable_attention_plots()

    seed_dir = os.path.join(output_root, f"seed_{seed}")
    attention_dir = os.path.join(seed_dir, "attention")
    cnnlstm_dir = os.path.join(seed_dir, "model_comparison")
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(cnnlstm_dir, exist_ok=True)

    print(f"Running fixed-window seed experiment: seed={seed}, window={window_size}/{stride}")
    print(f"Output root: {seed_dir}")

    summary_rows = []

    if "AttentionNet" in models:
        attention_exp.run_pipeline_with_attention(data_folder=data_folder, output_dir=attention_dir)
        attention_csv = os.path.join(attention_dir, "fold_level_metrics.csv")
        summary_rows.append({"model": "AttentionNet", **summarize_predictions(attention_csv)})

    if "CNNLSTM" in models:
        comparison_exp.RESULTS_DIR = cnnlstm_dir
        comparison_exp.run_ablation_experiment(
            selected_models=["CNNLSTM"],
            data_folder=data_folder,
        )
        cnnlstm_csv = os.path.join(cnnlstm_dir, "cnnlstm_subject_predictions.csv")
        summary_rows.append({"model": "CNNLSTM", **summarize_predictions(cnnlstm_csv)})

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(seed_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\nFixed-window seed summary:")
    print(summary.to_string(index=False))
    print(f"Summary saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-window LOSO seed sensitivity experiment.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--data-folder", default=attention_exp.DATA_FOLDER)
    parser.add_argument(
        "--output-root",
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "results",
                "seed_sensitivity_fixed_1000_500",
            )
        ),
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["AttentionNet", "CNNLSTM"],
        default=["AttentionNet"],
        help="Models to run. Defaults to AttentionNet only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        seed=args.seed,
        data_folder=args.data_folder,
        output_root=args.output_root,
        window_size=args.window_size,
        stride=args.stride,
        models=args.models,
    )
