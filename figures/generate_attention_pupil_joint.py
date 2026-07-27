#!/usr/bin/env python3
"""
Generate attention weight + pupil diameter joint timeline figure.
Shows what physiological events the attention mechanism focuses on.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common_paths import DATA_DIR, RESULTS_DIR, OUTPUT_DIR as FIGURE_OUTPUT_DIR

# Paths
DATA_FILE = str(DATA_DIR / 'labeled_12.csv')
ATTN_FILE = str(RESULTS_DIR / 'attention' / 'attention_weights_pid_1.npy')
OUTPUT_DIR = str(FIGURE_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
FS = 60  # Hz
DT = 1000 / FS  # ms per sample
WIN_LEN = 59  # frames per window
STRIDE = 30   # frames stride

COLOR_ATTN = '#C73E1D'
COLOR_PUPIL = '#2E86AB'


def reconstruct_attention_timeline(trial_len, attn_weights, win_len=WIN_LEN, stride=STRIDE):
    """Reconstruct per-sample attention by averaging overlapping windows."""
    n_win = attn_weights.shape[0]
    attn_t = np.zeros(trial_len)
    count_t = np.zeros(trial_len)
    
    for w_idx in range(n_win):
        start = w_idx * stride
        for j in range(win_len):
            t = start + j
            if t < trial_len:
                attn_t[t] += attn_weights[w_idx, j]
                count_t[t] += 1
    
    # Avoid division by zero
    attn_t[count_t > 0] /= count_t[count_t > 0]
    return attn_t


def main():
    # Load trial data (PID 1, Trial001 from labeled_12.csv)
    df = pd.read_csv(DATA_FILE, low_memory=False)
    trial_df = df[(df['ParticipantID'] == 1) & (df['Trial'] == 'Trial001')].copy()
    trial_df = trial_df.sort_values('RecordingTime [ms]').reset_index(drop=True)
    
    # Extract pupil diameter
    pupil = trial_df['Pupil Diameter Right [mm]'].replace('-', np.nan).astype(float).values
    time_ms = trial_df['RecordingTime [ms]'].values
    time_ms = time_ms - time_ms[0]  # Relative time from trial start
    
    # Load attention weights
    attn = np.load(ATTN_FILE)  # (n_windows, 59)
    
    # Trial001 produces ~69 windows; assume they map to the first 69 attention weights
    n_win_trial = (len(pupil) - WIN_LEN) // STRIDE + 1
    attn_trial = attn[:n_win_trial, :]
    
    # Reconstruct attention timeline
    attn_t = reconstruct_attention_timeline(len(pupil), attn_trial)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})
    
    ax1 = axes[0]
    ax1.fill_between(time_ms, attn_t, alpha=0.3, color=COLOR_ATTN)
    ax1.plot(time_ms, attn_t, color=COLOR_ATTN, linewidth=1.2, label='Attention weight')
    ax1.set_ylabel('Attention Weight', fontsize=11, color=COLOR_ATTN)
    ax1.tick_params(axis='y', labelcolor=COLOR_ATTN)
    ax1.set_title('(a) Attention Weight Timeline', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Mark attention peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(attn_t, height=np.nanpercentile(attn_t, 85), distance=FS*2)
    for p in peaks[:5]:
        ax1.axvline(time_ms[p], color=COLOR_ATTN, linestyle='--', alpha=0.4, linewidth=0.8)
    
    ax2 = axes[1]
    ax2.plot(time_ms, pupil, color=COLOR_PUPIL, linewidth=1.2, label='Pupil diameter')
    ax2.set_ylabel('Pupil Diameter (mm)', fontsize=11, color=COLOR_PUPIL)
    ax2.tick_params(axis='y', labelcolor=COLOR_PUPIL)
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_title('(b) Pupil Diameter (Right)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    # Mark attention peaks on pupil plot too
    for p in peaks[:5]:
        ax2.axvline(time_ms[p], color=COLOR_ATTN, linestyle='--', alpha=0.4, linewidth=0.8)
    
    plt.suptitle('Attention-Pupil Joint Timeline (PID 1, ASD, Trial001)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_pupil_joint.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_pupil_joint.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: attention_pupil_joint.png/pdf")
    print(f"  Trial length: {len(pupil)} samples ({len(pupil)/FS:.1f}s)")
    print(f"  Windows used: {n_win_trial}")
    print(f"  Attention peaks marked: {len(peaks)}")


if __name__ == '__main__':
    main()
