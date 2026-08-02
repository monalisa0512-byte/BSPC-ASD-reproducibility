#!/usr/bin/env python3
"""Generate the fixed-versus-nested multi-seed manuscript figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("AttentionNet", "CNNLSTM", "PureLSTM")
SEEDS = (42, 43, 44, 45, 46)
FULL = "Full_Preprocessing"
NO_FILTER = "No_Filtering"


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-aggregate",
        default=repository / "results/fixed_multiseed_strict/aggregate",
        type=Path,
    )
    parser.add_argument(
        "--nested-aggregate",
        default=repository / "results/nested_multiseed_strict/aggregate",
        type=Path,
    )
    parser.add_argument("--output-dir", default=repository / "figures/generated", type=Path)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "Arial", "serif"],
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def export_source_data(
    output: Path,
    fixed_metrics: pd.DataFrame,
    nested_metrics: pd.DataFrame,
    window_frequency: pd.DataFrame,
    selected_windows: pd.DataFrame,
) -> None:
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    panel_a = pd.concat(
        [
            fixed_metrics.assign(protocol="Fixed 1000/500 ms"),
            nested_metrics.assign(protocol="Nested selection"),
        ],
        ignore_index=True,
    )
    panel_a = panel_a[(panel_a["config"] == FULL) & panel_a["model"].isin(MODELS)]
    panel_a[["protocol", "model", "global_seed", "accuracy", "f1"]].to_csv(
        source / "protocol_sensitivity_panel_a.csv", index=False, encoding="utf-8-sig"
    )
    panel_b = pd.concat(
        [
            fixed_metrics[fixed_metrics["model"].eq("AttentionNet")]
            .pivot(index="global_seed", columns="config", values="accuracy")
            .assign(protocol="Fixed 1000/500 ms"),
            nested_metrics[nested_metrics["model"].eq("AttentionNet")]
            .pivot(index="global_seed", columns="config", values="accuracy")
            .assign(protocol="Nested selection"),
        ]
    ).reset_index()
    panel_b["full_minus_no_filtering"] = panel_b[FULL] - panel_b[NO_FILTER]
    panel_b.to_csv(source / "protocol_sensitivity_panel_b.csv", index=False, encoding="utf-8-sig")
    tie = (
        selected_windows.assign(tied=selected_windows["primary_score_tie_count"].gt(1))
        .groupby(["model", "config"], as_index=False)["tied"]
        .mean()
        .rename(columns={"tied": "tied_fold_fraction"})
    )
    window_frequency.merge(tie, on=["model", "config"], how="left").to_csv(
        source / "protocol_sensitivity_panel_c.csv", index=False, encoding="utf-8-sig"
    )


def main() -> int:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed_metrics = pd.read_csv(args.fixed_aggregate / "multiseed_task_metrics.csv")
    fixed_paired = pd.read_csv(args.fixed_aggregate / "multiseed_paired_differences.csv")
    nested_metrics = pd.read_csv(args.nested_aggregate / "nested_task_metrics.csv")
    nested_paired = pd.read_csv(args.nested_aggregate / "nested_paired_comparisons.csv")
    window_frequency = pd.read_csv(args.nested_aggregate / "nested_window_selection_frequency.csv")
    selected_windows = pd.read_csv(args.nested_aggregate / "nested_selected_window_by_fold.csv")
    export_source_data(args.output_dir, fixed_metrics, nested_metrics, window_frequency, selected_windows)

    protocol_colors = {"Fixed": "#4C78A8", "Nested": "#E28E2C"}
    seed_colors = dict(zip(SEEDS, ("#3B6FB6", "#7A5195", "#2A9D8F", "#D17C29", "#B84A62")))
    window_colors = {
        (1000, 500): "#4C78A8",
        (1500, 750): "#72B7B2",
        (2000, 500): "#F2CF5B",
        (2000, 1000): "#B9B9B9",
    }

    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0), height_ratios=(1.0, 1.0), hspace=0.55, wspace=0.38)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    # Panel a: paired fixed/nested accuracies for Full Preprocessing.
    offsets = {"Fixed": -0.18, "Nested": 0.18}
    for model_index, model in enumerate(MODELS):
        fixed = fixed_metrics[(fixed_metrics.model == model) & (fixed_metrics.config == FULL)].set_index("global_seed")
        nested = nested_metrics[(nested_metrics.model == model) & (nested_metrics.config == FULL)].set_index("global_seed")
        for seed in SEEDS:
            y_fixed = 100 * fixed.loc[seed, "accuracy"]
            y_nested = 100 * nested.loc[seed, "accuracy"]
            ax_a.plot(
                [model_index + offsets["Fixed"], model_index + offsets["Nested"]],
                [y_fixed, y_nested],
                color=seed_colors[seed], alpha=0.45, linewidth=0.75, zorder=1,
            )
            ax_a.scatter(model_index + offsets["Fixed"], y_fixed, s=17, facecolor="white", edgecolor=seed_colors[seed], linewidth=0.8, zorder=2)
            ax_a.scatter(model_index + offsets["Nested"], y_nested, s=17, marker="s", facecolor=seed_colors[seed], edgecolor="white", linewidth=0.4, zorder=2)
        for protocol, frame in (("Fixed", fixed), ("Nested", nested)):
            values = 100 * frame.loc[list(SEEDS), "accuracy"].to_numpy()
            x = model_index + offsets[protocol]
            ax_a.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt="D", ms=4.2,
                          color=protocol_colors[protocol], markeredgecolor="black", markeredgewidth=0.45,
                          capsize=2.5, linewidth=1.15, zorder=4)
    ax_a.set_xticks(range(len(MODELS)), ("AttentionNet", "CNN-LSTM", "Pure LSTM"))
    ax_a.set_ylabel("Participant-level accuracy (%)")
    ax_a.set_ylim(63, 87)
    ax_a.set_title("Full Preprocessing across protocols", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    fixed_handle = mpl.lines.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=protocol_colors["Fixed"], linestyle="none", label="Fixed")
    nested_handle = mpl.lines.Line2D([], [], marker="s", markerfacecolor=protocol_colors["Nested"], markeredgecolor="white", linestyle="none", label="Nested")
    ax_a.legend(handles=[fixed_handle, nested_handle], loc="lower right", ncol=2, handletextpad=0.4, columnspacing=0.8)

    # Panel b: Full minus No Filtering for AttentionNet.
    protocol_frames = []
    for label, frame in (("Fixed", fixed_metrics), ("Nested", nested_metrics)):
        pivot = frame[frame.model.eq("AttentionNet")].pivot(index="global_seed", columns="config", values="accuracy")
        delta = 100 * (pivot[FULL] - pivot[NO_FILTER])
        protocol_frames.append((label, delta))
    for seed in SEEDS:
        ax_b.plot([0, 1], [protocol_frames[0][1].loc[seed], protocol_frames[1][1].loc[seed]], color=seed_colors[seed], alpha=0.55, linewidth=0.8)
        ax_b.scatter(0, protocol_frames[0][1].loc[seed], s=17, facecolor="white", edgecolor=seed_colors[seed], linewidth=0.8, zorder=2)
        ax_b.scatter(1, protocol_frames[1][1].loc[seed], s=17, marker="s", facecolor=seed_colors[seed], edgecolor="white", linewidth=0.4, zorder=2)
    fixed_row = fixed_paired[(fixed_paired.model == "AttentionNet") & fixed_paired.comparison.str.contains("No_Filtering")].iloc[0]
    nested_row = nested_paired[nested_paired.comparison.str.startswith("AttentionNet Full_Preprocessing minus No_Filtering")].iloc[0]
    means = [100 * fixed_row.delta_accuracy_mean, 100 * nested_row.delta_accuracy_mean]
    lows = [100 * fixed_row.crossed_bootstrap_delta_accuracy_ci_low, 100 * nested_row.accuracy_ci95_low]
    highs = [100 * fixed_row.crossed_bootstrap_delta_accuracy_ci_high, 100 * nested_row.accuracy_ci95_high]
    for x, mean, low, high, color in zip((0, 1), means, lows, highs, protocol_colors.values()):
        ax_b.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="D", color=color,
                      markeredgecolor="black", markeredgewidth=0.45, capsize=2.5, linewidth=1.2, zorder=4)
    ax_b.axhline(0, color="#444444", linestyle="--", linewidth=0.7)
    ax_b.set_xticks((0, 1), ("Fixed", "Nested"))
    ax_b.set_ylabel("Full − No Filtering (pp)")
    ax_b.set_ylim(-12, 13)
    ax_b.set_title("AttentionNet filtering contrast", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color="#E5E5E5", linewidth=0.55)

    # Panel c: nested window-selection frequencies and tie prevalence.
    groups = (
        ("AttentionNet\nFull", "AttentionNet", FULL),
        ("AttentionNet\nNo Filtering", "AttentionNet", NO_FILTER),
        ("CNN-LSTM\nFull", "CNNLSTM", FULL),
        ("Pure LSTM\nFull", "PureLSTM", FULL),
    )
    y_positions = np.arange(len(groups))
    left = np.zeros(len(groups))
    for window in ((1000, 500), (1500, 750), (2000, 500), (2000, 1000)):
        values = []
        for _, model, config in groups:
            row = window_frequency[(window_frequency.model == model) & (window_frequency.config == config) &
                                   (window_frequency.window_ms == window[0]) & (window_frequency.stride_ms == window[1])]
            values.append(float(row.iloc[0].percent))
        bars = ax_c.barh(y_positions, values, left=left, height=0.62, color=window_colors[window],
                         edgecolor="white", linewidth=0.35, label=f"{window[0]}/{window[1]}")
        for bar, value, start in zip(bars, values, left):
            if value >= 8:
                ax_c.text(start + value / 2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}",
                          ha="center", va="center", fontsize=5.8, color="white" if window == (1000, 500) else "#222222")
        left += np.asarray(values)
    tie_rates = []
    for _, model, config in groups:
        subset = selected_windows[(selected_windows.model == model) & (selected_windows.config == config)]
        tie_rates.append(100 * subset.primary_score_tie_count.gt(1).mean())
    for y, rate in zip(y_positions, tie_rates):
        ax_c.text(101.5, y, f"tie {rate:.0f}%", va="center", ha="left", fontsize=6.2)
    ax_c.set_yticks(y_positions, [item[0] for item in groups])
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 116)
    ax_c.set_xlabel("Selected-window frequency (%)")
    ax_c.set_title("Nested window selection", loc="left", fontweight="bold")
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2, columnspacing=0.8, handlelength=1.2)

    for label, axis in zip(("a", "b", "c"), (ax_a, ax_b, ax_c)):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.16)
    base = args.output_dir / "protocol_sensitivity"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    contract = {
        "core_conclusion": "Model, preprocessing, and window-selection results vary across seeds and protocols; no displayed contrast has an interval excluding zero.",
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "font": "Times New Roman",
        "independent_unit": "participant (n=57)",
        "pipeline_seeds": list(SEEDS),
        "panels": {
            "a": "Full-Preprocessing accuracy by model, protocol, and seed",
            "b": "AttentionNet Full-minus-No-Filtering paired seed differences and crossed-bootstrap intervals",
            "c": "Nested window-selection frequencies and validation-tie prevalence",
        },
    }
    (args.output_dir / "protocol_sensitivity_figure_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved protocol sensitivity figure to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
