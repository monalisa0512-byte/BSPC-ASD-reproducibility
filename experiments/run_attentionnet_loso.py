r"""
ASD eye-tracking classification pipeline (clean 57-subject LOSO)
WITH attention extraction and fold-level variance analysis.
- Uses data/eyesdata_processed_57 by default, or BSPC_DATA_DIR when set.
- No extra training PIDs; strict 57-fold LOSO only.
- Saves attention curves, fold-level metrics CSV, and variance plots.
"""
import argparse
import os
from common_paths import DATA_DIR as DEFAULT_DATA_DIR
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
import copy
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.interpolate import PchipInterpolator
from collections import Counter

warnings.filterwarnings("ignore")

# =========================
# 0. Environment & GPU
# =========================
print("=" * 40)
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print("=" * 40)

# =========================
# 1. Global Config
# =========================
DATA_FOLDER = str(DEFAULT_DATA_DIR)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "attention"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 1000
STRIDE = 500
WINDOW_CANDIDATES = [(2000, 1000), (1500, 750), (1000, 500), (2000, 500)]
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50

MAX_TRIAL_MISSING_RATE = 0.80
MAX_MISSING_RATE = 0.60
MAX_CONTINUOUS_FRAMES = 60

PRE_MS = 80
POST_MS = 160
SAMPLING_MS = 16.67

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BASE_FEATURES = [
    "Tracking Ratio [%]", "Pupil Diameter Right [mm]", "Pupil Diameter Left [mm]",
    "Point of Regard Right X [px]", "Point of Regard Right Y [px]",
    "Point of Regard Left X [px]",  "Point of Regard Left Y [px]"
]
MASK_FEATURES = [col + "_Mask" for col in BASE_FEATURES]
ALL_FEATURES = BASE_FEATURES + MASK_FEATURES

PID_COL, Y_COL, TIME_COL, TRIAL_COL = "Participant", "Class", "RecordingTime [ms]", "Stimulus"

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# =========================
# 2. Preprocessing helpers
# =========================
def check_mask_distribution(full_df):
    print("\n" + "=" * 50)
    print("Mask missing rate distribution")
    print("=" * 50)
    col_mask = "Pupil Diameter Right [mm]_Mask"
    if col_mask not in full_df.columns:
        print("=" * 50 + "\n")
        return
    grouped = full_df.groupby(Y_COL)
    for name, group in grouped:
        total_rows = len(group)
        missing_rows = group[col_mask].sum()
        missing_rate = (missing_rows / total_rows) * 100 if total_rows > 0 else 0
        print(f"Class {name}: total {total_rows} rows, missing {int(missing_rows)} rows ({missing_rate:.2f}%)")
    print("=" * 50 + "\n")


def expand_missing_mask(arr_mask_bool, pre_ms=80, post_ms=160, sampling_ms=16.67):
    pre = int(round(pre_ms / sampling_ms))
    post = int(round(post_ms / sampling_ms))
    mask = arr_mask_bool.astype(bool).copy()
    padded = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            s = max(0, i - pre)
            e = min(n, j + post)
            padded[s:e] = True
            i = j
        else:
            i += 1
    return padded


def pchip_blockwise_impute(df_block, base_features, limit_ffill=5, limit_bfill=5):
    out = df_block.copy()
    for col in base_features:
        arr = out[col].values.astype(float)
        arr[arr == 0] = np.nan
        isnan = np.isnan(arr)

        if (~isnan).sum() >= 2:
            idx = np.arange(len(arr))
            valid_idx = idx[~isnan]
            valid_vals = arr[~isnan]
            f = PchipInterpolator(valid_idx, valid_vals)
            interp_vals = f(idx)
            first, last = valid_idx[0], valid_idx[-1]
            arr[first:last+1] = interp_vals[first:last+1]

        s = pd.Series(arr)
        s = s.fillna(method='ffill', limit=limit_ffill)
        s = s.fillna(method='bfill', limit=limit_bfill)

        if s.isna().any():
            median_val = np.nanmedian(out[col].values)
            if np.isnan(median_val):
                median_val = 0.0
            s = s.fillna(median_val)

        out[col] = s.values
    return out


def load_all_data(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "labeled_*.csv"))
    if not all_files:
        raise ValueError(f"No CSV files found in {folder_path}")

    print(f"Loading {len(all_files)} files ...")
    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f, low_memory=False)
            if 'ParticipantID' in temp_df.columns:
                if PID_COL not in temp_df.columns:
                    temp_df = temp_df.rename(columns={'ParticipantID': PID_COL})
                else:
                    temp_df = temp_df.drop(columns=['ParticipantID'])
            if 'Trial' in temp_df.columns:
                if TRIAL_COL not in temp_df.columns:
                    temp_df = temp_df.rename(columns={'Trial': TRIAL_COL})
                else:
                    temp_df = temp_df.drop(columns=['Trial'])
            if temp_df.columns.duplicated().any():
                temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
            if TIME_COL in temp_df.columns:
                temp_df = temp_df.sort_values(TIME_COL)

            for col in BASE_FEATURES:
                if col not in temp_df.columns:
                    temp_df[col] = np.nan
                else:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

                is_missing = temp_df[col].isna() | (temp_df[col] == 0)
                expanded_mask = expand_missing_mask(
                    is_missing.values, pre_ms=PRE_MS, post_ms=POST_MS, sampling_ms=SAMPLING_MS
                )
                temp_df[col + "_Mask"] = expanded_mask.astype(float)
                temp_df.loc[expanded_mask, col] = np.nan

            blocks = [
                g for _, g in temp_df.groupby(
                    (temp_df[TRIAL_COL] != temp_df[TRIAL_COL].shift()).cumsum()
                )
            ]
            recon_blocks = []
            for b in blocks:
                trial_mask = b[MASK_FEATURES].values.astype(float)
                frame_missing_rate = trial_mask.mean(axis=1)
                severe_missing = (frame_missing_rate >= 0.5).astype(int)
                trial_missing_rate = severe_missing.mean()
                if trial_missing_rate > MAX_TRIAL_MISSING_RATE:
                    continue
                recon = pchip_blockwise_impute(b, BASE_FEATURES, limit_ffill=5, limit_bfill=5)
                for mcol in MASK_FEATURES:
                    recon[mcol] = b[mcol].values
                recon_blocks.append(recon)

            if not recon_blocks:
                continue

            temp_df = pd.concat(recon_blocks, ignore_index=True)
            if PID_COL in temp_df.columns:
                temp_df = temp_df.dropna(subset=[PID_COL])
                try:
                    temp_df[PID_COL] = temp_df[PID_COL].astype(int)
                except Exception:
                    temp_df[PID_COL] = pd.to_numeric(temp_df[PID_COL], errors='coerce').fillna(0).astype(int)
                df_list.append(temp_df)
        except Exception as e:
            print(f"[WARN] File {f} failed: {e}")

    if not df_list:
        raise ValueError("No usable files after preprocessing")
    return pd.concat(df_list, ignore_index=True)


# =========================
# 3. Windowing (with start indices for time alignment)
# =========================
def filter_nan_windows_with_pid(X_list, Y_list, P_list):
    if not X_list:
        return np.array([]), np.array([]), np.array([])
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    P_arr = np.array(P_list)
    valid_mask = ~np.isnan(X_arr).any(axis=(1, 2))
    return X_arr[valid_mask], Y_arr[valid_mask], P_arr[valid_mask]


def filter_nan_windows(X_list, Y_list):
    if not X_list:
        return np.array([]), np.array([])
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    valid_mask = ~np.isnan(X_arr).any(axis=(1, 2))
    return X_arr[valid_mask], Y_arr[valid_mask]


def get_max_continuous_missing(mask_1d):
    padded = np.pad(mask_1d, (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return np.max(ends - starts)


def get_window_rows(window_size_ms, stride_ms):
    return int(window_size_ms / SAMPLING_MS), int(stride_ms / SAMPLING_MS)


def window_passes_quality(current_win):
    current_mask = current_win[:, len(BASE_FEATURES):]
    frame_missing_rate = current_mask.mean(axis=1)
    severe_missing = (frame_missing_rate >= 0.5).astype(int)
    missing_rate = severe_missing.mean()
    max_continuous = get_max_continuous_missing(severe_missing)
    return missing_rate <= MAX_MISSING_RATE and max_continuous <= MAX_CONTINUOUS_FRAMES


def count_valid_windows_for_pids(full_df, target_pids, window_size_ms, stride_ms):
    """Window-quality diagnostic used only on inner-fold subjects."""
    target_pids = set(target_pids)
    if not target_pids:
        return {}

    win_rows, stride_rows = get_window_rows(window_size_ms, stride_ms)
    counts = Counter()
    for pid, group in full_df.groupby(PID_COL):
        if pid not in target_pids:
            continue
        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()
        for _, block_data in group.groupby(block_ids):
            raw_base = block_data[BASE_FEATURES].values.astype(float)
            raw_mask = block_data[MASK_FEATURES].values.astype(float)
            if len(raw_base) < win_rows:
                continue

            raw_combined = np.concatenate([raw_base, raw_mask], axis=1)
            for start in range(0, len(raw_combined) - win_rows + 1, stride_rows):
                if window_passes_quality(raw_combined[start: start + win_rows]):
                    counts[pid] += 1
    return counts


def select_window_config_from_inner_fold(full_df, inner_pids, candidates=WINDOW_CANDIDATES):
    """Select window parameters without using the held-out test subject."""
    if len(candidates) == 1:
        return candidates[0]

    ranked = []
    for window_size_ms, stride_ms in candidates:
        counts = count_valid_windows_for_pids(full_df, inner_pids, window_size_ms, stride_ms)
        covered_subjects = sum(1 for c in counts.values() if c > 0)
        median_windows = np.median(list(counts.values())) if counts else 0
        ranked.append((covered_subjects, median_windows, -window_size_ms, -stride_ms, window_size_ms, stride_ms))

    best = max(ranked)
    return best[-2], best[-1]


def create_dataset_from_full_data_with_starts(
    full_df, train_pids, test_pid, scaler=None, window_size_ms=WINDOW_SIZE, stride_ms=STRIDE
):
    """Returns train/test data AND test window start indices for temporal alignment."""
    X_train, Y_train, P_train = [], [], []
    X_test, Y_test, starts_test = [], [], []

    win_rows, stride_rows = get_window_rows(window_size_ms, stride_ms)

    grouped = full_df.groupby(PID_COL)
    for pid, group in grouped:
        label = 1.0 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0.0
        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()

        for _, block_data in group.groupby(block_ids):
            raw_base = block_data[BASE_FEATURES].values.astype(float)
            raw_mask = block_data[MASK_FEATURES].values.astype(float)
            if len(raw_base) < win_rows:
                continue

            raw_combined = np.concatenate([raw_base, raw_mask], axis=1)
            for start in range(0, len(raw_combined) - win_rows + 1, stride_rows):
                current_win = raw_combined[start: start + win_rows]

                if window_passes_quality(current_win):
                    if pid == test_pid:
                        X_test.append(current_win)
                        Y_test.append(label)
                        starts_test.append(start)
                    elif pid in train_pids:
                        X_train.append(current_win)
                        Y_train.append(label)
                        P_train.append(pid)

    X_train_arr, Y_train_arr, P_train_arr = filter_nan_windows_with_pid(X_train, Y_train, P_train)
    X_test_arr, Y_test_arr = filter_nan_windows(X_test, Y_test)
    starts_test = np.array(starts_test, dtype=np.int32)
    # Keep only starts corresponding to valid test windows
    valid_test = ~np.isnan(np.array(X_test, dtype=np.float32)).any(axis=(1, 2))
    starts_test = starts_test[valid_test]

    if len(X_train_arr) == 0 or len(X_test_arr) == 0:
        return None, None, None, None, None, None, None

    train_base_feats_for_fit = X_train_arr[:, :, :7].reshape(-1, 7)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_base_feats_for_fit)

    N_tr, T, F_total = X_train_arr.shape
    N_te = X_test_arr.shape[0]

    X_train_base_scaled = scaler.transform(
        X_train_arr[:, :, :7].reshape(-1, 7)
    ).reshape(N_tr, T, 7)
    X_train_mask = X_train_arr[:, :, 7:]
    X_test_base_scaled = scaler.transform(
        X_test_arr[:, :, :7].reshape(-1, 7)
    ).reshape(N_te, T, 7)
    X_test_mask = X_test_arr[:, :, 7:]

    X_train_final = np.concatenate([X_train_base_scaled, X_train_mask], axis=2)
    X_test_final = np.concatenate([X_test_base_scaled, X_test_mask], axis=2)

    return X_train_final, Y_train_arr, P_train_arr, X_test_final, Y_test_arr, starts_test, scaler


def extract_single_pid_data_pure(
    full_df, target_pid, scaler, window_size_ms=WINDOW_SIZE, stride_ms=STRIDE
):
    grouped = full_df.groupby(PID_COL)
    X_pid, Y_pid = [], []
    win_rows, stride_rows = get_window_rows(window_size_ms, stride_ms)

    for pid, group in grouped:
        if pid != target_pid:
            continue
        label = 1.0 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0.0

        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()
        for _, block_data in group.groupby(block_ids):
            raw_base = block_data[BASE_FEATURES].values.astype(float)
            raw_mask = block_data[MASK_FEATURES].values.astype(float)
            if len(raw_base) < win_rows:
                continue

            raw_combined = np.concatenate([raw_base, raw_mask], axis=1)
            for start in range(0, len(raw_combined) - win_rows + 1, stride_rows):
                current_win = raw_combined[start: start + win_rows]

                if window_passes_quality(current_win):
                    X_pid.append(current_win)
                    Y_pid.append(label)

    X_pid_arr, Y_pid_arr = filter_nan_windows(X_pid, Y_pid)
    if len(X_pid_arr) == 0:
        return None, None

    N, T, F_total = X_pid_arr.shape
    X_pid_base_scaled = scaler.transform(
        X_pid_arr[:, :, :7].reshape(-1, 7)
    ).reshape(N, T, 7)
    X_pid_mask = X_pid_arr[:, :, 7:]
    X_pid_final = np.concatenate([X_pid_base_scaled, X_pid_mask], axis=2)

    return X_pid_final, Y_pid_arr


# =========================
# 4. Model (with attention extraction)
# =========================
def get_stratified_train_val_pids(pool_pids, pool_labels, test_size=0.15, random_state=None):
    pool_pids = list(pool_pids)
    pool_labels = list(pool_labels)
    n = len(pool_pids)

    if n <= 6:
        val_count = max(1, int(round(n * 0.15)))
        val_pids = np.random.choice(pool_pids, size=val_count, replace=False).tolist()
        real_train_pids = [p for p in pool_pids if p not in val_pids]
        return real_train_pids, val_pids

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, val_idx in sss.split(np.arange(n), pool_labels):
            real_train_pids = [pool_pids[idx] for idx in train_idx]
            val_pids = [pool_pids[idx] for idx in val_idx]

        val_labels = [pool_labels[pool_pids.index(v)] for v in val_pids]
        if (0 in val_labels) and (1 in val_labels):
            return real_train_pids, val_pids

        for cls in (0, 1):
            if cls not in val_labels:
                candidates = [
                    p for p, l in zip(pool_pids, pool_labels)
                    if l == cls and p in real_train_pids
                ]
                if candidates:
                    chosen = candidates[0]
                    real_train_pids.remove(chosen)
                    val_pids.append(chosen)
                    break
        return real_train_pids, val_pids
    except Exception:
        val_count = max(1, int(round(n * 0.15)))
        val_pids = np.random.choice(pool_pids, size=val_count, replace=False).tolist()
        real_train_pids = [p for p in pool_pids if p not in val_pids]
        return real_train_pids, val_pids


class AttentionNet(nn.Module):
    def __init__(self, input_dim):
        super(AttentionNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.attn_fc = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_fc(out), dim=1)
        return self.fc(self.dropout(torch.sum(attn_weights * out, dim=1))).squeeze(-1)

    def forward_with_attention(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        attn_logits = self.attn_fc(out)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pred = self.fc(self.dropout(torch.sum(attn_weights * out, dim=1))).squeeze(-1)
        return pred, attn_weights.squeeze(-1)  # (batch, seq_len)


# =========================
# 5. Attention plotting
# =========================
def plot_attention_for_subject(attn_matrix, starts, pid, label_name, save_path):
    """
    attn_matrix: (n_windows, 60) numpy array
    starts: (n_windows,) frame start indices
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2]})

    # Top: individual window attention curves (lightly transparent)
    ax0 = axes[0]
    t_window_ms = np.arange(attn_matrix.shape[1]) * (1000 / 60)
    for i in range(min(attn_matrix.shape[0], 50)):
        offset_ms = starts[i] * (1000 / 60)
        abs_time = t_window_ms + offset_ms
        ax0.plot(abs_time, attn_matrix[i], alpha=0.2, color="steelblue")
    ax0.set_ylabel("Attention weight")
    ax0.set_title(f"PID {pid} ({label_name}) — Overlapping window attentions")
    ax0.set_ylim(bottom=0)

    # Bottom: time-aligned average attention
    ax1 = axes[1]
    max_len = int(starts.max() + attn_matrix.shape[1]) + 1
    agg_attn = np.zeros(max_len)
    counts = np.zeros(max_len)
    for i in range(attn_matrix.shape[0]):
        s = starts[i]
        e = s + attn_matrix.shape[1]
        agg_attn[s:e] += attn_matrix[i]
        counts[s:e] += 1
    avg_attn = np.divide(agg_attn, counts, out=np.zeros_like(agg_attn), where=counts > 0)
    valid = counts > 0
    t_ms = np.arange(max_len) * (1000 / 60)

    ax1.fill_between(t_ms[valid], avg_attn[valid], alpha=0.4, color="coral")
    ax1.plot(t_ms[valid], avg_attn[valid], color="darkred", linewidth=1.5)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Averaged attention")
    ax1.set_title("Time-aligned average attention")
    ax1.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Attention figure saved: {save_path}")


# =========================
# 6. Main training + attention extraction
# =========================
def run_pipeline_with_attention(data_folder=DATA_FOLDER, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    full_df = load_all_data(data_folder)
    check_mask_distribution(full_df)

    pid_to_label = {}
    for pid, group in full_df.groupby(PID_COL):
        pid_to_label[pid] = 1 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0

    all_pids = sorted(pid_to_label.keys())
    print(f"Total subjects: {len(all_pids)}")
    print(f"LOSO folds: {len(all_pids)}")

    y_true_all, y_pred_all = [], []
    scaler_amp = torch.cuda.amp.GradScaler()

    # Fold-level metrics storage
    fold_records = []

    for i, test_pid in enumerate(all_pids):
        print(f"\n=== Fold {i+1}/{len(all_pids)} | Test PID: {test_pid} ===")
        pool_pids = [p for p in all_pids if p != test_pid]
        pool_labels = [pid_to_label[p] for p in pool_pids]

        real_train_pids, val_pids = get_stratified_train_val_pids(
            pool_pids, pool_labels, test_size=0.15, random_state=SEED + i
        )

        selected_candidate, selected_score = None, None
        for window_size_ms, stride_ms in WINDOW_CANDIDATES:
            print(f"  Candidate window -> size: {window_size_ms} ms | stride: {stride_ms} ms")

            out = create_dataset_from_full_data_with_starts(
                full_df, real_train_pids, test_pid,
                window_size_ms=window_size_ms, stride_ms=stride_ms
            )
            if out[0] is None or len(out[0]) == 0:
                print("    Skipping candidate: no valid train/test windows.")
                continue
            X_tr, Y_tr, P_tr, X_te, Y_te, starts_te, scaler = out

            X_val_list = []
            for v_pid in val_pids:
                x_v, y_v = extract_single_pid_data_pure(
                    full_df, v_pid, scaler,
                    window_size_ms=window_size_ms, stride_ms=stride_ms
                )
                if x_v is not None and len(x_v) > 0:
                    label_val = int(np.round(np.mean(y_v)))
                    X_val_list.append(
                        (torch.tensor(x_v, dtype=torch.float32).to(DEVICE), label_val)
                    )

            if len(X_val_list) == 0:
                print("    Skipping candidate: empty validation set.")
                continue

            # Subject-balanced sampler
            pid_counts = Counter(P_tr)
            class_0_pids = set(p for p, y in zip(P_tr, Y_tr) if y == 0)
            class_1_pids = set(p for p, y in zip(P_tr, Y_tr) if y == 1)
            n_class_0_pids = max(1, len(class_0_pids))
            n_class_1_pids = max(1, len(class_1_pids))

            sample_weights = []
            for p, y in zip(P_tr, Y_tr):
                n_pids_in_class = n_class_1_pids if y == 1 else n_class_0_pids
                weight = 1.0 / (pid_counts[p] * n_pids_in_class)
                sample_weights.append(weight)

            sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

            print(
                f"    Split -> Train: {len(real_train_pids)} subjects "
                f"(Samples:{len(X_tr)}) | Val: {len(X_val_list)} subjects"
            )

            t_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
            t_Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(DEVICE)

            train_loader = DataLoader(
                TensorDataset(t_X_tr, t_Y_tr), batch_size=BATCH_SIZE, sampler=sampler
            )

            model = AttentionNet(input_dim=X_tr.shape[2]).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
            )
            criterion = nn.BCEWithLogitsLoss()

            best_val_metric, best_val_f1, best_val_acc = -1.0, 0.0, 0.0
            best_thresh, best_model_weights = 0.5, None
            patience, patience_counter = 12, 0

            epoch_pbar = tqdm(
                range(EPOCHS),
                desc=f"Fold {i+1} {window_size_ms}/{stride_ms}",
                leave=False
            )
            for epoch in epoch_pbar:
                model.train()
                for bx, by in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast():
                        loss = criterion(model(bx), by)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    scheduler.step()

                model.eval()
                val_probs_epoch, val_labels_epoch = [], []
                with torch.no_grad():
                    for t_X_v, y_v in X_val_list:
                        with torch.cuda.amp.autocast():
                            probs = torch.sigmoid(model(t_X_v)).cpu().numpy()
                        val_probs_epoch.append(np.median(probs))
                        val_labels_epoch.append(y_v)

                current_best_j, current_best_th = -1.0, 0.5
                current_best_preds = None
                for th in np.arange(0.30, 0.71, 0.01):
                    preds = [1 if p > th else 0 for p in val_probs_epoch]
                    tn, fp, fn, tp = confusion_matrix(val_labels_epoch, preds, labels=[0, 1]).ravel()
                    sens = tp / (tp + fn + 1e-7)
                    spec = tn / (tn + fp + 1e-7)
                    j_stat = sens + spec - 1
                    if j_stat > current_best_j:
                        current_best_j, current_best_th = j_stat, th
                        current_best_preds = preds

                current_val_acc = accuracy_score(val_labels_epoch, current_best_preds)
                current_val_f1 = f1_score(val_labels_epoch, current_best_preds)
                if current_best_j > best_val_metric:
                    best_val_metric, best_thresh = current_best_j, current_best_th
                    best_val_acc, best_val_f1 = current_val_acc, current_val_f1
                    best_model_weights = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            candidate_score = (
                best_val_metric, best_val_f1, best_val_acc, -window_size_ms, -stride_ms
            )
            print(
                f"    Val J: {best_val_metric:.4f} | Val F1: {best_val_f1:.4f} | "
                f"Val Acc: {best_val_acc:.4f} | Threshold: {best_thresh:.2f}"
            )
            if best_model_weights is not None and (
                selected_score is None or candidate_score > selected_score
            ):
                selected_score = candidate_score
                selected_candidate = {
                    "window_size_ms": window_size_ms,
                    "stride_ms": stride_ms,
                    "weights": best_model_weights,
                    "input_dim": X_tr.shape[2],
                    "X_te": X_te,
                    "Y_te": Y_te,
                    "starts_te": starts_te,
                    "best_thresh": best_thresh,
                    "val_j": best_val_metric,
                    "val_f1": best_val_f1,
                    "val_acc": best_val_acc,
                }

            del model, optimizer, scheduler, train_loader, t_X_tr, t_Y_tr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if selected_candidate is None:
            print(f"  Error: no valid nested window candidate, skipping Fold {i+1}.")
            continue

        window_size_ms = selected_candidate["window_size_ms"]
        stride_ms = selected_candidate["stride_ms"]
        best_thresh = selected_candidate["best_thresh"]
        X_te = selected_candidate["X_te"]
        Y_te = selected_candidate["Y_te"]
        starts_te = selected_candidate["starts_te"]
        t_X_te = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

        model = AttentionNet(input_dim=selected_candidate["input_dim"]).to(DEVICE)
        model.load_state_dict(selected_candidate["weights"])
        model.eval()
        print(
            f"  Selected window -> {window_size_ms}/{stride_ms} ms "
            f"(Val J={selected_candidate['val_j']:.4f}, "
            f"Val F1={selected_candidate['val_f1']:.4f})"
        )

        # --- Extract test probabilities ---
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                test_probs = torch.sigmoid(model(t_X_te)).cpu().numpy()

        final_score = np.median(test_probs)
        test_pred = 1 if final_score > best_thresh else 0
        y_true_all.append(int(Y_te[0]))
        y_pred_all.append(test_pred)

        # --- Fold-level metrics ---
        tn, fp, fn, tp = 0, 0, 0, 0
        if int(Y_te[0]) == 1 and test_pred == 1:
            tp = 1
        elif int(Y_te[0]) == 0 and test_pred == 1:
            fp = 1
        elif int(Y_te[0]) == 1 and test_pred == 0:
            fn = 1
        elif int(Y_te[0]) == 0 and test_pred == 0:
            tn = 1

        fold_acc = 1.0 if test_pred == int(Y_te[0]) else 0.0
        fold_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fold_f1 = 1.0 if (test_pred == 1 and int(Y_te[0]) == 1) else 0.0

        fold_records.append({
            "fold": i + 1,
            "pid": test_pid,
            "true_label": int(Y_te[0]),
            "pred": test_pred,
            "accuracy": fold_acc,
            "f1": fold_f1,
            "sensitivity": fold_sens,
            "specificity": fold_spec,
            "window_size_ms": window_size_ms,
            "stride_ms": stride_ms,
            "inner_val_j": selected_candidate["val_j"],
            "inner_val_f1": selected_candidate["val_f1"],
            "inner_val_acc": selected_candidate["val_acc"],
            "threshold": best_thresh,
            "median_prob": final_score,
        })

        print(
            f"  Best threshold: {best_thresh:.2f} | "
            f"True: {int(Y_te[0])}, Median prob: {final_score:.4f}, Pred: {test_pred}"
        )

        # --- Extract and save attention for this test subject ---
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _, attn_weights = model.forward_with_attention(t_X_te)
        attn_np = attn_weights.cpu().numpy()  # (n_windows, 60)

        label_name = "ASD" if int(Y_te[0]) == 1 else "TD"

        # Save raw attention weights for group-level analysis
        npy_path = os.path.join(output_dir, f"attention_weights_pid_{test_pid}.npy")
        np.save(npy_path, attn_np)

        fig_path = os.path.join(output_dir, f"attention_pid_{test_pid}_{label_name}.png")
        plot_attention_for_subject(attn_np, starts_te, test_pid, label_name, fig_path)

    # =========================
    # Final summary
    # =========================
    final_acc = accuracy_score(y_true_all, y_pred_all)
    final_f1 = f1_score(y_true_all, y_pred_all)

    print("\n" + "=" * 50)
    print("Bootstrap 95% CI...")
    boot_accs, boot_f1s = [], []
    for _ in range(1000):
        boot_y_true, boot_y_pred = resample(y_true_all, y_pred_all)
        boot_accs.append(accuracy_score(boot_y_true, boot_y_pred))
        boot_f1s.append(f1_score(boot_y_true, boot_y_pred))

    print("\n" + "=" * 50)
    print("Final Results")
    print("=" * 50)
    print(
        f"Accuracy : {final_acc*100:.2f}%  "
        f"(95% CI: {np.percentile(boot_accs, 2.5)*100:.2f}% - {np.percentile(boot_accs, 97.5)*100:.2f}%)"
    )
    print(
        f"F1 Score : {final_f1*100:.2f}%  "
        f"(95% CI: {np.percentile(boot_f1s, 2.5)*100:.2f}% - {np.percentile(boot_f1s, 97.5)*100:.2f}%)"
    )

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true_all, y_pred_all, display_labels=["TD", "ASD"], cmap=plt.cm.Blues, ax=ax
    )
    plt.title(f"CM (Acc: {final_acc*100:.1f}%)")
    cm_path = os.path.join(output_dir, "confusion_matrix_loso.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"\nConfusion matrix saved: {cm_path}")
    plt.close()

    # =========================
    # Fold-level summary table + plot
    # =========================
    df_folds = pd.DataFrame(fold_records)
    csv_path = os.path.join(output_dir, "fold_level_metrics.csv")
    df_folds.to_csv(csv_path, index=False)
    print(f"Fold-level metrics saved: {csv_path}")

    print("\n" + "=" * 50)
    print("Fold-level Performance Summary (n=57)")
    print("=" * 50)
    for col in ["accuracy", "f1", "sensitivity", "specificity", "threshold"]:
        vals = df_folds[col].values
        print(f"{col.capitalize():12s}: {vals.mean()*100:.2f}% ± {vals.std()*100:.2f}% "
              f"[min={vals.min()*100:.2f}%, max={vals.max()*100:.2f}%]")

    # Boxplot of fold-level accuracies and thresholds
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df_folds["accuracy"].values, ax=axes[0], color="steelblue")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Fold-level Accuracy Distribution")

    sns.boxplot(y=df_folds["threshold"].values, ax=axes[1], color="coral")
    axes[1].set_ylabel("Threshold")
    axes[1].set_title("Optimized Threshold Distribution")
    plt.tight_layout()
    box_path = os.path.join(output_dir, "fold_level_variance.png")
    plt.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Fold-level variance plot saved: {box_path}")

    return final_acc, final_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Run final AttentionNet LOSO pipeline.")
    parser.add_argument("--data-folder", default=DATA_FOLDER, help="Directory containing cleaned participant CSV files.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for fold metrics and attention outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline_with_attention(data_folder=args.data_folder, output_dir=args.output_dir)
