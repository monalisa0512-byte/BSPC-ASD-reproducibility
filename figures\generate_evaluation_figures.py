#!/usr/bin/env python3
"""
Generate P1 Evaluation figures for BSPC paper.
Figures:
  2a: LOSO 57-fold classification heatmap
  2b: Youden's J threshold optimization curves
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from common_paths import DATA_DIR as DATA_PATH, RESULTS_DIR, OUTPUT_DIR as FIGURE_OUTPUT_DIR

# Paths
DATA_DIR = str(DATA_PATH)
OUTPUT_DIR = str(FIGURE_OUTPUT_DIR)
FOLD_METRICS = str(RESULTS_DIR / 'attention' / 'fold_level_metrics.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
COLOR_ASD = '#C73E1D'
COLOR_TD = '#2E86AB'
COLOR_PRIMARY = '#2E86AB'
COLOR_SECONDARY = '#A23B72'
COLOR_CORRECT = '#4ECDC4'
COLOR_WRONG = '#FF6B6B'


def load_metadata():
    """Load participant metadata from processed data."""
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('labeled_') and f.endswith('.csv')])
    records = []
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)
        for pid in df['ParticipantID'].dropna().unique():
            pid_df = df[df['ParticipantID'] == pid]
            records.append({
                'ParticipantID': int(pid),
                'Gender': pid_df['Gender'].iloc[0] if not pid_df['Gender'].isna().all() else None,
                'Class': pid_df['Class'].iloc[0] if not pid_df['Class'].isna().all() else None,
            })
    df_meta = pd.DataFrame(records).drop_duplicates('ParticipantID')
    return df_meta


def plot_2a_loso_heatmap(df_fold, df_meta):
    """Figure 2a: LOSO 57-fold classification heatmap.

    Rows = participants (sorted by class: ASD then TD)
    Cols = fold number (1-57)
    Each cell = classification result when this participant is the TEST subject in this fold.
    Since LOSO has exactly one test subject per fold, each column has exactly one colored cell.
    """
    # Merge with metadata to get class
    df_fold = df_fold.merge(df_meta[['ParticipantID', 'Class']], left_on='pid', right_on='ParticipantID', how='left')

    # Sort by class (ASD first, then TD), then by PID
    df_fold = df_fold.sort_values(['Class', 'pid'], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(14, 8))

    # Build the heatmap matrix
    # Rows = participants (sorted), Cols = folds
    pids = sorted(df_meta[df_meta['Class'] == 'ASD']['ParticipantID'].tolist()) + \
           sorted(df_meta[df_meta['Class'] == 'TD']['ParticipantID'].tolist())

    n_pids = len(pids)
    n_folds = len(df_fold)

    # Initialize with NaN (gray)
    mat = np.full((n_pids, n_folds), np.nan)

    for _, row in df_fold.iterrows():
        fold_idx = int(row['fold']) - 1
        pid = int(row['pid'])
        pid_idx = pids.index(pid)
        true = int(row['true_label'])
        pred = int(row['pred'])
        correct = 1 if true == pred else 0
        mat[pid_idx, fold_idx] = correct

    # Custom colormap: red=wrong, teal=correct
    cmap = plt.cm.colors.ListedColormap([COLOR_WRONG, COLOR_CORRECT])

    masked_mat = np.ma.masked_where(np.isnan(mat), mat)
    im = ax.imshow(masked_mat, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    # Fill NaN background
    nan_mask = np.isnan(mat)
    ax.imshow(nan_mask, aspect='auto', cmap=plt.cm.colors.ListedColormap(['none', '#F0F0F0']),
              vmin=0, vmax=1, alpha=0.5)

    # Add fold separator lines every 10 folds
    for i in range(1, n_folds // 10 + 1):
        ax.axvline(x=i * 10 - 0.5, color='white', linewidth=0.5, alpha=0.5)

    # Class separator line
    n_asd = len(df_meta[df_meta['Class'] == 'ASD'])
    ax.axhline(y=n_asd - 0.5, color='black', linewidth=1.5, linestyle='--')
    ax.text(-3, n_asd / 2 - 0.5, 'ASD', va='center', ha='right', fontsize=10,
            fontweight='bold', color=COLOR_ASD, rotation=90)
    ax.text(-3, n_asd + (n_pids - n_asd) / 2 - 0.5, 'TD', va='center', ha='right', fontsize=10,
            fontweight='bold', color=COLOR_TD, rotation=90)

    # Mark misclassified subjects
    misclassified_pids = []
    for i, pid in enumerate(pids):
        row_data = df_fold[df_fold['pid'] == pid]
        if len(row_data) > 0 and row_data.iloc[0]['pred'] != row_data.iloc[0]['true_label']:
            misclassified_pids.append(pid)
            # Mark with a black border or symbol
            fold_idx = int(row_data.iloc[0]['fold']) - 1
            ax.add_patch(plt.Rectangle((fold_idx - 0.4, i - 0.4), 0.8, 0.8,
                         fill=False, edgecolor='black', linewidth=2))
            ax.text(fold_idx + 0.5, i + 0.5, f'S{pid}', fontsize=6, color='black',
                    fontweight='bold', ha='center', va='center')

    # Set ticks
    ax.set_xticks(np.arange(0, n_folds, 5))
    ax.set_xticklabels([str(i+1) for i in range(0, n_folds, 5)], fontsize=8)
    ax.set_yticks(np.arange(n_pids))
    ax.set_yticklabels([f'S {int(p)}' for p in pids], fontsize=7)

    # Legend
    red_patch = plt.matplotlib.patches.Patch(color=COLOR_WRONG, label='Misclassified')
    teal_patch = plt.matplotlib.patches.Patch(color=COLOR_CORRECT, label='Correct')
    ax.legend(handles=[red_patch, teal_patch], loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=9)

    ax.set_title('LOSO 57-Fold Classification Results\n(One test subject per fold; misclassified subjects highlighted)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('LOSO Fold', fontsize=11)
    ax.set_ylabel('Participant (test subject)', fontsize=11)

    # Summary text
    correct_count = df_fold['pred'].eq(df_fold['true_label']).sum()
    total = len(df_fold)
    ax.text(1.02, 0.02, f'Overall: {correct_count}/{total} correct ({correct_count/total*100:.1f}%)',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loso_57x57_heatmap.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'loso_57x57_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: loso_57x57_heatmap.pdf/png")


def plot_2b_youden_curve(df_fold):
    """Figure 2b: Youden's J threshold optimization.
    Panel (a): Threshold distribution across 57 folds
    Panel (b): Representative J curve computed from all 57 subject-level median probabilities
    """
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # --- Panel (a): Optimal threshold distribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = df_fold['threshold'].values

    ax1.hist(thresholds, bins=20, color=COLOR_PRIMARY, edgecolor='white', alpha=0.7)
    ax1.axvline(thresholds.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {thresholds.mean():.3f}')
    ax1.axvline(np.median(thresholds), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(thresholds):.3f}')

    # Annotate range
    ax1.annotate(f'Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=9, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel('Optimal Threshold (τ)', fontsize=10)
    ax1.set_ylabel('Number of Folds', fontsize=10)
    ax1.set_title('(a) Per-Fold Optimal Threshold Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # --- Panel (b): Representative J curve ---
    ax2 = fig.add_subplot(gs[0, 1])

    # Use all 57 subject-level median probabilities to compute a global J curve
    probs = df_fold['median_prob'].values
    true_labels = df_fold['true_label'].values.astype(int)

    # Compute J at different thresholds
    tau_range = np.arange(0.10, 0.91, 0.01)
    sensitivities = []
    specificities = []
    j_values = []

    n_asd = true_labels.sum()
    n_td = len(true_labels) - n_asd

    for tau in tau_range:
        pred = (probs >= tau).astype(int)
        tp = ((pred == 1) & (true_labels == 1)).sum()
        fn = ((pred == 0) & (true_labels == 1)).sum()
        tn = ((pred == 0) & (true_labels == 0)).sum()
        fp = ((pred == 1) & (true_labels == 0)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1

        sensitivities.append(sens)
        specificities.append(spec)
        j_values.append(j)

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    j_values = np.array(j_values)

    # Plot curves
    ax2.plot(tau_range, sensitivities, color=COLOR_ASD, linewidth=2, label='Sensitivity', linestyle='-')
    ax2.plot(tau_range, specificities, color=COLOR_TD, linewidth=2, label='Specificity', linestyle='-')
    ax2.plot(tau_range, j_values, color='green', linewidth=2.5, label="Youden's J", linestyle='-')

    # Mark optimal point
    best_idx = np.argmax(j_values)
    best_tau = tau_range[best_idx]
    best_j = j_values[best_idx]
    ax2.scatter([best_tau], [best_j], color='green', s=100, zorder=5, edgecolor='black', linewidth=1.5)
    ax2.annotate(f'J* = {best_j:.3f}\nτ* = {best_tau:.2f}',
                 xy=(best_tau, best_j), xytext=(best_tau + 0.08, best_j - 0.05),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Mark per-fold mean threshold
    mean_tau = thresholds.mean()
    ax2.axvline(mean_tau, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax2.text(mean_tau + 0.01, 0.15, f'Mean fold\nτ={mean_tau:.2f}', fontsize=7, color='red')

    ax2.set_xlabel('Threshold (τ)', fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_title("(b) Youden's J Optimization Curve\n(Global, n=57 subject-level median probs)",
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, loc='center right')
    ax2.set_xlim(0.10, 0.90)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.suptitle('Threshold Optimization Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'youden_j_optimization.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'youden_j_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: youden_j_optimization.pdf/png")


if __name__ == '__main__':
    print("Loading fold-level metrics...")
    df_fold = pd.read_csv(FOLD_METRICS)
    print(f"  Folds: {len(df_fold)}")

    print("Loading participant metadata...")
    df_meta = load_metadata()
    print(f"  Participants: {len(df_meta)}")

    print("\nGenerating Figure 2a: LOSO 57x57 heatmap...")
    plot_2a_loso_heatmap(df_fold, df_meta)

    print("\nGenerating Figure 2b: Youden's J curve...")
    plot_2b_youden_curve(df_fold)

    print("\nAll P1 figures generated successfully!")
