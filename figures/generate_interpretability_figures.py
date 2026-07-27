#!/usr/bin/env python3
"""
Generate P3 Interpretability figures for BSPC paper.
Figures:
  3b: ASD vs TD attention entropy density distribution
  3c: Layer-wise t-SNE (CNN output -> LSTM output -> Attention context vector)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from common_paths import DATA_DIR as DATA_PATH, RESULTS_DIR, OUTPUT_DIR as FIGURE_OUTPUT_DIR

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

# Optional: t-SNE
try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, t-SNE figure will be skipped.")

# Paths
DATA_DIR = str(DATA_PATH)
ATTN_DIR = str(RESULTS_DIR / 'attention')
OUTPUT_DIR = str(FIGURE_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
COLOR_ASD = '#C73E1D'
COLOR_TD = '#2E86AB'


def load_metadata():
    """Load participant metadata."""
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('labeled_') and f.endswith('.csv')])
    records = []
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)
        for pid in df['ParticipantID'].dropna().unique():
            pid_df = df[df['ParticipantID'] == pid]
            records.append({
                'ParticipantID': int(pid),
                'Class': pid_df['Class'].iloc[0] if not pid_df['Class'].isna().all() else None,
            })
    df_meta = pd.DataFrame(records).drop_duplicates('ParticipantID')
    return df_meta


def compute_attention_entropy(attn_weights):
    """Compute entropy for each attention window.
    attn_weights: (n_windows, window_length) array
    Returns: (n_windows,) array of entropies
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    attn = attn_weights + eps
    # Normalize to sum to 1 (they may already be softmaxed, but just in case)
    attn = attn / attn.sum(axis=1, keepdims=True)
    entropy = -np.sum(attn * np.log(attn), axis=1)
    return entropy


def plot_3b_attention_entropy(df_meta):
    """Figure 3b: ASD vs TD attention entropy density distribution."""
    # Load all attention weights
    asd_entropies = []
    td_entropies = []

    for _, row in df_meta.iterrows():
        pid = int(row['ParticipantID'])
        cls = row['Class']
        attn_file = os.path.join(ATTN_DIR, f'attention_weights_pid_{pid}.npy')
        if not os.path.exists(attn_file):
            print(f"  Warning: attention weights not found for PID {pid}")
            continue

        attn = np.load(attn_file)
        entropies = compute_attention_entropy(attn)
        mean_entropy = entropies.mean()

        if cls == 'ASD':
            asd_entropies.append(mean_entropy)
        elif cls == 'TD':
            td_entropies.append(mean_entropy)

    asd_entropies = np.array(asd_entropies)
    td_entropies = np.array(td_entropies)

    # Mann-Whitney U test
    if len(asd_entropies) > 0 and len(td_entropies) > 0:
        statistic, pvalue = stats.mannwhitneyu(asd_entropies, td_entropies, alternative='two-sided')
    else:
        statistic, pvalue = np.nan, np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel (a): KDE density ---
    ax1 = axes[0]

    # Compute KDE
    x_range = np.linspace(min(asd_entropies.min(), td_entropies.min()) - 0.05,
                          max(asd_entropies.max(), td_entropies.max()) + 0.05, 500)

    if len(asd_entropies) >= 2:
        kde_asd = stats.gaussian_kde(asd_entropies)
        ax1.plot(x_range, kde_asd(x_range), color=COLOR_ASD, linewidth=2.5, label=f'ASD (n={len(asd_entropies)})')
        ax1.fill_between(x_range, kde_asd(x_range), alpha=0.2, color=COLOR_ASD)
    if len(td_entropies) >= 2:
        kde_td = stats.gaussian_kde(td_entropies)
        ax1.plot(x_range, kde_td(x_range), color=COLOR_TD, linewidth=2.5, label=f'TD (n={len(td_entropies)})')
        ax1.fill_between(x_range, kde_td(x_range), alpha=0.2, color=COLOR_TD)

    # Mean lines
    ax1.axvline(asd_entropies.mean(), color=COLOR_ASD, linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(td_entropies.mean(), color=COLOR_TD, linestyle='--', linewidth=1.5, alpha=0.7)

    # Annotation
    ax1.text(0.98, 0.95, f'Mann-Whitney U: p={pvalue:.4f}\n'
                          f'ASD μ={asd_entropies.mean():.3f}\n'
                          f'TD μ={td_entropies.mean():.3f}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel('Mean Attention Entropy (bits)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('(a) Attention Entropy Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # --- Panel (b): Box plot + swarm ---
    ax2 = axes[1]

    bp = ax2.boxplot([asd_entropies, td_entropies], labels=['ASD', 'TD'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(COLOR_ASD)
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(COLOR_TD)
    bp['boxes'][1].set_alpha(0.5)

    # Add individual points
    np.random.seed(42)
    jitter_asd = np.random.normal(1, 0.04, size=len(asd_entropies))
    jitter_td = np.random.normal(2, 0.04, size=len(td_entropies))
    ax2.scatter(jitter_asd, asd_entropies, color=COLOR_ASD, alpha=0.6, s=30, zorder=3)
    ax2.scatter(jitter_td, td_entropies, color=COLOR_TD, alpha=0.6, s=30, zorder=3)

    ax2.set_ylabel('Mean Attention Entropy (bits)', fontsize=11)
    ax2.set_title('(b) Per-Subject Attention Entropy', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Attention Entropy: ASD vs TD', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_entropy_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_entropy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: attention_entropy_distribution.pdf/png")
    print(f"  ASD n={len(asd_entropies)}, TD n={len(td_entropies)}, p={pvalue:.4f}")


def plot_3c_tsne_placeholder():
    """Figure 3c: Layer-wise t-SNE visualization.

    Note: This requires intermediate features (CNN output, LSTM output, Attention context vector)
    which are not cached in the current pipeline. A full forward pass with feature extraction
    would need to be run. For now, generate a placeholder or skip.
    """
    print("Skipping layer-wise t-SNE: intermediate CNN/LSTM/context features are not cached by the current pipeline.")
    print("No placeholder or simulated t-SNE figure is generated.")


if __name__ == '__main__':
    print("Loading participant metadata...")
    df_meta = load_metadata()
    print(f"  Participants: {len(df_meta)}")

    print("\nGenerating Figure 3b: Attention entropy distribution...")
    plot_3b_attention_entropy(df_meta)

    print("\nGenerating Figure 3c: Layer-wise t-SNE...")
    plot_3c_tsne_placeholder()

    print("\nAll P3 figures generated!")
    if not SKLEARN_AVAILABLE:
        print("  (t-SNE skipped due to missing sklearn)")
    print("  (t-SNE skipped unless real intermediate features are exported by the model pipeline)")
