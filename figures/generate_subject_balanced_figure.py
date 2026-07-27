"""
Generate subject-balanced sampling evidence figure.
Shows raw window count imbalance and equalized effective contribution.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from common_paths import DATA_DIR, RESULTS_DIR, OUTPUT_DIR as FIGURE_OUTPUT_DIR

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9

OUTPUT_DIR = str(FIGURE_OUTPUT_DIR)
COUNTS_PATH = str(RESULTS_DIR / "pid_window_counts.json")

COLOR_TD = '#2E86AB'
COLOR_ASD = '#C73E1D'


def generate_subject_balanced_figure():
    with open(COUNTS_PATH, 'r') as f:
        pid_window_counts = json.load(f)

    # Load labels from processed data (same 57 PIDs)
    import glob
    DATA_FOLDER = str(DATA_DIR)
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    df_list = []
    for fpath in all_files:
        df = pd.read_csv(fpath, low_memory=False)
        if 'ParticipantID' in df.columns and 'Class' in df.columns:
            df_list.append(df[['ParticipantID', 'Class']].drop_duplicates())
    labels_df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['ParticipantID'])
    labels_df = labels_df[~labels_df['ParticipantID'].isin([12, 16])]
    pid_to_label = dict(zip(labels_df['ParticipantID'].astype(int), labels_df['Class']))

    records = []
    for pid_str, count in pid_window_counts.items():
        pid = int(pid_str)
        label = pid_to_label.get(pid, 'Unknown')
        records.append({'PID': pid, 'Class': label, 'Windows': count})

    df = pd.DataFrame(records)
    df = df.sort_values(['Class', 'Windows'], ascending=[True, False])

    # Effective weight per participant under subject-balanced sampling
    n_asd = (df['Class'] == 'ASD').sum()
    n_td = (df['Class'] == 'TD').sum()
    df['EffectiveProb'] = df.apply(
        lambda row: 1.0 / (2 * n_asd) if row['Class'] == 'ASD' else 1.0 / (2 * n_td), axis=1
    )
    df['RawProb'] = df['Windows'] / df['Windows'].sum()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Raw window counts
    ax = axes[0]
    asd_df = df[df['Class'] == 'ASD'].sort_values('Windows', ascending=True)
    td_df = df[df['Class'] == 'TD'].sort_values('Windows', ascending=True)

    y_pos_asd = np.arange(len(asd_df))
    y_pos_td = np.arange(len(td_df)) + len(asd_df) + 1  # gap between groups

    ax.barh(y_pos_asd, asd_df['Windows'], color=COLOR_ASD, height=0.7, alpha=0.85, label=f'ASD (n={n_asd})')
    ax.barh(y_pos_td, td_df['Windows'], color=COLOR_TD, height=0.7, alpha=0.85, label=f'TD (n={n_td})')

    # Annotations
    ax.axhline(y=len(asd_df) - 0.5 + 0.5, color='#888888', linestyle='--', lw=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, len(asd_df) / 2, 'ASD', ha='right', va='center', fontsize=10, color=COLOR_ASD, fontweight='bold')
    ax.text(ax.get_xlim()[1] * 0.98, len(asd_df) + 1 + len(td_df) / 2, 'TD', ha='right', va='center', fontsize=10, color=COLOR_TD, fontweight='bold')

    ax.set_xlabel('Valid Windows per Participant', fontweight='bold')
    ax.set_ylabel('Participant (sorted by window count)', fontweight='bold')
    ax.set_title('(a) Raw Window Count Distribution', fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_yticks([])

    # Add median lines
    ax.axvline(x=asd_df['Windows'].median(), color=COLOR_ASD, linestyle=':', lw=1.2, alpha=0.7)
    ax.axvline(x=td_df['Windows'].median(), color=COLOR_TD, linestyle=':', lw=1.2, alpha=0.7)
    ax.text(asd_df['Windows'].median() + 20, 2, f"ASD median\n{asd_df['Windows'].median():.0f}", fontsize=7, color=COLOR_ASD, va='top')
    ax.text(td_df['Windows'].median() + 20, len(asd_df) + 3, f"TD median\n{td_df['Windows'].median():.0f}", fontsize=7, color=COLOR_TD, va='top')

    # Right: Effective sampling probability
    ax = axes[1]
    asd_df_sorted = df[df['Class'] == 'ASD'].sort_values('RawProb', ascending=True)
    td_df_sorted = df[df['Class'] == 'TD'].sort_values('RawProb', ascending=True)

    y_pos_asd = np.arange(len(asd_df_sorted))
    y_pos_td = np.arange(len(td_df_sorted)) + len(asd_df_sorted) + 1

    ax.barh(y_pos_asd, asd_df_sorted['RawProb'] * 100, color=COLOR_ASD, height=0.7, alpha=0.4, label='Raw proportion')
    ax.barh(y_pos_td, td_df_sorted['RawProb'] * 100, color=COLOR_TD, height=0.7, alpha=0.4)

    # Overlay effective probabilities as scatter or horizontal lines
    eff_asd = asd_df_sorted['EffectiveProb'].iloc[0] * 100
    eff_td = td_df_sorted['EffectiveProb'].iloc[0] * 100

    ax.scatter([eff_asd] * len(asd_df_sorted), y_pos_asd, color=COLOR_ASD, s=30, zorder=5, marker='D', label='Subject-balanced')
    ax.scatter([eff_td] * len(td_df_sorted), y_pos_td, color=COLOR_TD, s=30, zorder=5, marker='D')

    # Reference lines
    ax.axvline(x=eff_asd, color=COLOR_ASD, linestyle='--', lw=1.2, alpha=0.7)
    ax.axvline(x=eff_td, color=COLOR_TD, linestyle='--', lw=1.2, alpha=0.7)

    ax.axhline(y=len(asd_df_sorted) - 0.5 + 0.5, color='#888888', linestyle='--', lw=0.8, alpha=0.5)

    ax.set_xlabel('Contribution to Training Batches (%)', fontweight='bold')
    ax.set_ylabel('')
    ax.set_title('(b) Effective Contribution: Raw vs. Balanced', fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_yticks([])

    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_ASD, alpha=0.85, label=f'ASD (n={n_asd})'),
        Patch(facecolor=COLOR_TD, alpha=0.85, label=f'TD (n={n_td})'),
        Patch(facecolor='gray', alpha=0.4, label='Raw proportion'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=8, label='Subject-balanced')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_subject_balanced_sampling.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_subject_balanced_sampling.pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: fig_subject_balanced_sampling.png/pdf")
    print(f"  ASD: median {asd_df['Windows'].median():.0f}, range [{asd_df['Windows'].min():.0f}, {asd_df['Windows'].max():.0f}]")
    print(f"  TD:  median {td_df['Windows'].median():.0f}, range [{td_df['Windows'].min():.0f}, {td_df['Windows'].max():.0f}]")


if __name__ == '__main__':
    generate_subject_balanced_figure()
