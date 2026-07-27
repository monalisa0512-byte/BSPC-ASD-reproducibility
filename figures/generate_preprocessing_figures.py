#!/usr/bin/env python3
"""
Generate P2 Preprocessing figures for BSPC paper.
Figures:
  1a: PCHIP vs Linear vs Cubic Spline interpolation comparison
  1c: Blink boundary artifact zoom (with/without expanded mask)
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from common_paths import RAW_DIR as RAW_PATH, OUTPUT_DIR as FIGURE_OUTPUT_DIR

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

# Paths
RAW_DIR = str(RAW_PATH)
OUTPUT_DIR = str(FIGURE_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
COLOR_PCHIP = '#2E86AB'
COLOR_LINEAR = '#A23B72'
COLOR_SPLINE = '#F18F01'
COLOR_RAW = '#888888'
COLOR_MASK = '#E74C3C'

# Sampling parameters
FS = 60  # Hz
DT = 1000 / FS  # ms per sample
PRE_EXTEND_MS = 80
POST_EXTEND_MS = 160
PRE_FRAMES = int(round(PRE_EXTEND_MS / DT))  # ~5
POST_FRAMES = int(round(POST_EXTEND_MS / DT))  # ~10


def find_good_segment(filepath, target_missing=30, window_samples=600):
    """Find a window with a good missing run for interpolation demo."""
    df = pd.read_csv(filepath, low_memory=False)
    pd_col = df['Pupil Diameter Right [mm]'].replace('-', np.nan).astype(float)
    valid = pd_col.notna()

    # Find missing runs
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(valid):
        if not v and not in_run:
            in_run = True
            start = i
        elif v and in_run:
            in_run = False
            runs.append((start, i, i - start))
    if in_run:
        runs.append((start, len(valid), len(valid) - start))

    # Find a window containing a run of appropriate length (20-150 samples)
    good_runs = [r for r in runs if 20 <= r[2] <= 150]
    if not good_runs:
        return None, None, None

    # Pick a run that's not too close to edges
    for s, e, l in sorted(good_runs, key=lambda x: abs(x[2] - target_missing)):
        win_start = max(0, s - 200)
        win_end = min(len(df), win_start + window_samples)
        win_start = win_end - window_samples
        if win_start >= 0:
            return df, win_start, win_end

    return None, None, None


def apply_blink_mask(valid_array, pre=PRE_FRAMES, post=POST_FRAMES):
    """Expand blink mask backward by pre frames and forward by post frames."""
    mask = ~valid_array.copy()
    n = len(mask)
    expanded = mask.copy()
    for i in range(n):
        if mask[i]:
            start = max(0, i - pre)
            end = min(n, i + post + 1)
            expanded[start:end] = True
    return expanded


def interpolate_methods(t, y, mask):
    """Apply three interpolation methods to masked data.
    t: time array (ms)
    y: pupil diameter values (NaN where masked)
    mask: boolean mask (True = missing/interpolate)
    Returns: dict of method_name -> interpolated values
    """
    valid_idx = ~mask
    t_valid = t[valid_idx]
    y_valid = y[valid_idx]

    results = {}

    # PCHIP
    pchip = interpolate.PchipInterpolator(t_valid, y_valid)
    results['PCHIP'] = pchip(t)

    # Linear
    linear = interpolate.interp1d(t_valid, y_valid, kind='linear', fill_value='extrapolate')
    results['Linear'] = linear(t)

    # Cubic Spline (may oscillate)
    try:
        cs = interpolate.CubicSpline(t_valid, y_valid)
        results['Cubic Spline'] = cs(t)
    except ValueError:
        # Not enough points for CubicSpline
        results['Cubic Spline'] = linear(t)

    return results


def plot_1a_interpolation_comparison():
    """Figure 1a: PCHIP vs Linear vs Cubic Spline on a real 10s segment."""
    # Find a good file and segment
    files = ['14.csv', '16.csv', '13.csv', '15.csv']
    for f in files:
        filepath = os.path.join(RAW_DIR, f)
        df, win_start, win_end = find_good_segment(filepath)
        if df is not None:
            break
    else:
        print("No suitable segment found!")
        return

    # Extract segment
    segment = df.iloc[win_start:win_end].reset_index(drop=True)
    pd_col = segment['Pupil Diameter Right [mm]'].replace('-', np.nan).astype(float)
    n = len(segment)

    # Time axis in ms
    t = np.arange(n) * DT

    # Raw values
    y_raw = pd_col.values.copy()
    valid = pd_col.notna().values

    # Apply expanded blink mask
    expanded_mask = apply_blink_mask(valid)

    # Create masked version (NaN in expanded mask regions)
    y_masked = y_raw.copy()
    y_masked[expanded_mask] = np.nan

    # Interpolate
    results = interpolate_methods(t, y_masked, expanded_mask)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1 = axes[0]
    # Raw signal (light gray, only valid points)
    ax1.plot(t[valid], y_raw[valid], 'o-', color=COLOR_RAW, alpha=0.3, markersize=2, linewidth=0.8, label='Raw valid')

    # Masked regions (shaded)
    mask_regions = []
    in_mask = False
    m_start = 0
    for i, m in enumerate(expanded_mask):
        if m and not in_mask:
            in_mask = True
            m_start = i
        elif not m and in_mask:
            in_mask = False
            mask_regions.append((m_start, i))
    if in_mask:
        mask_regions.append((m_start, len(expanded_mask)))

    for s, e in mask_regions:
        ax1.axvspan(t[s], t[e], alpha=0.15, color=COLOR_MASK)

    # Interpolated curves
    ax1.plot(t, results['PCHIP'], '-', color=COLOR_PCHIP, linewidth=2, label='PCHIP')
    ax1.plot(t, results['Linear'], '--', color=COLOR_LINEAR, linewidth=2, label='Linear')
    ax1.plot(t, results['Cubic Spline'], '-.', color=COLOR_SPLINE, linewidth=2, label='Cubic Spline')

    ax1.set_ylabel('Pupil Diameter (mm)', fontsize=11)
    ax1.set_title('(a) Interpolation Comparison on a 10 s Pupil Diameter Segment\n'
                  '(Shaded = expanded blink mask; pre=80 ms, post=160 ms)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)

    # Panel (b): Difference from PCHIP
    ax2 = axes[1]
    ax2.plot(t, results['Linear'] - results['PCHIP'], '--', color=COLOR_LINEAR, linewidth=1.5, label='Linear − PCHIP')
    ax2.plot(t, results['Cubic Spline'] - results['PCHIP'], '-.', color=COLOR_SPLINE, linewidth=1.5, label='Cubic Spline − PCHIP')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    for s, e in mask_regions:
        ax2.axvspan(t[s], t[e], alpha=0.15, color=COLOR_MASK)
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Difference (mm)', fontsize=10)
    ax2.set_title('(b) Deviation from PCHIP', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interpolation_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'interpolation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: interpolation_comparison.pdf/png")


def plot_1c_boundary_zoom():
    """Figure 1c: Zoom into blink boundary to show artifact differences."""
    # Use the same segment
    files = ['14.csv', '16.csv', '13.csv', '15.csv']
    for f in files:
        filepath = os.path.join(RAW_DIR, f)
        df, win_start, win_end = find_good_segment(filepath)
        if df is not None:
            break
    else:
        print("No suitable segment found!")
        return

    segment = df.iloc[win_start:win_end].reset_index(drop=True)
    pd_col = segment['Pupil Diameter Right [mm]'].replace('-', np.nan).astype(float)
    n = len(segment)
    t = np.arange(n) * DT
    y_raw = pd_col.values.copy()
    valid = pd_col.notna().values
    expanded_mask = apply_blink_mask(valid)
    y_masked = y_raw.copy()
    y_masked[expanded_mask] = np.nan

    results = interpolate_methods(t, y_masked, expanded_mask)

    # Find the first significant mask region for zoom
    mask_regions = []
    in_mask = False
    m_start = 0
    for i, m in enumerate(expanded_mask):
        if m and not in_mask:
            in_mask = True
            m_start = i
        elif not m and in_mask:
            in_mask = False
            mask_regions.append((m_start, i))
    if in_mask:
        mask_regions.append((m_start, len(expanded_mask)))

    # Pick a region that's not too long and has valid data on both sides
    good_region = None
    for s, e in mask_regions:
        if 10 <= e - s <= 100 and s > 20 and e < n - 20:
            good_region = (s, e)
            break
    if good_region is None:
        good_region = mask_regions[0] if mask_regions else (0, min(50, n))

    s, e = good_region
    zoom_start = max(0, s - 30)
    zoom_end = min(n, e + 30)

    t_zoom = t[zoom_start:zoom_end]
    y_raw_zoom = y_raw[zoom_start:zoom_end]
    valid_zoom = valid[zoom_start:zoom_end]
    mask_zoom = expanded_mask[zoom_start:zoom_end]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top row: with expanded mask
    # Bottom row: without expanded mask (only original missing)

    original_mask = ~valid_zoom

    for row_idx, (mask, title_suffix) in enumerate([(mask_zoom, 'with expanded mask'),
                                                      (original_mask, 'without expanded mask')]):
        ax = axes[row_idx, 0]

        # Valid raw points
        ax.plot(t_zoom[valid_zoom], y_raw_zoom[valid_zoom], 'o', color=COLOR_RAW, alpha=0.4, markersize=3)

        # Masked region
        mask_regions_zoom = []
        in_m = False
        ms = 0
        for i, m in enumerate(mask):
            if m and not in_m:
                in_m = True
                ms = i
            elif not m and in_m:
                in_m = False
                mask_regions_zoom.append((ms, i))
        if in_m:
            mask_regions_zoom.append((ms, len(mask)))

        for ms, me in mask_regions_zoom:
            ax.axvspan(t_zoom[ms], t_zoom[me], alpha=0.15, color=COLOR_MASK)

        # Interpolations for this mask
        y_zoom = y_raw_zoom.copy()
        y_zoom[mask] = np.nan
        res = interpolate_methods(t_zoom, y_zoom, mask)

        ax.plot(t_zoom, res['PCHIP'], '-', color=COLOR_PCHIP, linewidth=2.5, label='PCHIP')
        ax.plot(t_zoom, res['Linear'], '--', color=COLOR_LINEAR, linewidth=2, label='Linear')
        ax.plot(t_zoom, res['Cubic Spline'], '-.', color=COLOR_SPLINE, linewidth=2, label='Cubic Spline')

        ax.set_ylabel('Pupil Diameter (mm)', fontsize=10)
        ax.set_title(f'({"c" if row_idx == 0 else "d"}) Boundary {title_suffix}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Right column: zoomed difference
        ax2 = axes[row_idx, 1]
        ax2.plot(t_zoom, res['Linear'] - res['PCHIP'], '--', color=COLOR_LINEAR, linewidth=1.5, label='Linear − PCHIP')
        ax2.plot(t_zoom, res['Cubic Spline'] - res['PCHIP'], '-.', color=COLOR_SPLINE, linewidth=1.5, label='Cubic Spline − PCHIP')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        for ms, me in mask_regions_zoom:
            ax2.axvspan(t_zoom[ms], t_zoom[me], alpha=0.15, color=COLOR_MASK)
        ax2.set_ylabel('Difference (mm)', fontsize=10)
        ax2.set_title(f'({"e" if row_idx == 0 else "f"}) Deviation from PCHIP', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    plt.suptitle('Blink Boundary Interpolation Artifacts: Expanded Mask Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boundary_artifacts.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'boundary_artifacts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: boundary_artifacts.pdf/png")


if __name__ == '__main__':
    print("Generating Figure 1a: Interpolation comparison...")
    plot_1a_interpolation_comparison()

    print("\nGenerating Figure 1c: Boundary artifacts...")
    plot_1c_boundary_zoom()

    print("\nAll P2 figures generated successfully!")
