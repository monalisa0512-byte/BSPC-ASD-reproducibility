import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
# from tqdm.auto import tqdm
import warnings
import copy
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.interpolate import PchipInterpolator
from collections import Counter
import json
import sys
import time
import random

warnings.filterwarnings("ignore")

# =========================
# 0. 环境与 GPU 设置
# =========================
print("="*60)
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print("="*60)

# =========================
# 1. 全局配置
# =========================
DATA_FOLDER = os.environ.get(
    "BSPC_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "eyesdata_processed_57")),
)
RESULTS_FOLDER = os.environ.get(
    "BSPC_FIXED_RESULTS_FOLDER",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "attentionnet_preprocessing_ablation_pipeline_fixed")),
)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 设置日志文件
LOG_FILE = os.path.join(RESULTS_FOLDER, "attentionnet_preprocessing_ablation_pipeline_fixed_log.txt")

def log_print(message):
    """同时打印到控制台和日志文件"""
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

log_print(f"\n{'='*60}")
log_print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"{'='*60}")

WINDOW_SIZE = 1000
STRIDE = 500
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50
MAX_FOLDS = None

# Ultimate标准参数（测试集锁定）
ULTIMATE_MAX_TRIAL_MISSING_RATE = 0.80
ULTIMATE_MAX_MISSING_RATE = 0.60
ULTIMATE_MAX_CONTINUOUS_FRAMES = 60
ULTIMATE_PRE_MS = 80
ULTIMATE_POST_MS = 160
SAMPLING_MS = 16.67

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BASE_FEATURES = [
    "Tracking Ratio [%]", "Pupil Diameter Right [mm]", "Pupil Diameter Left [mm]",
    "Point of Regard Right X [px]", "Point of Regard Right Y [px]",
    "Point of Regard Left X [px]",  "Point of Regard Left Y [px]"
]
MASK_FEATURES = [col + "_Mask" for col in BASE_FEATURES]
PID_COL, Y_COL, TIME_COL, TRIAL_COL = "Participant", "Class", "RecordingTime [ms]", "Stimulus"

def seed_everything(seed):
    """Reset all process-level RNGs so every ablation starts identically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# =========================
# 2. 消融实验配置类
# =========================
class AblationConfig:
    def __init__(self, name, use_j_stat=True, pre_ms=80, post_ms=160,
                 impute_method='pchip', use_mask_features=True,
                 trial_intercept=True, window_intercept=True):
        self.name = name
        self.use_j_stat = use_j_stat
        self.pre_ms = pre_ms
        self.post_ms = post_ms
        self.impute_method = impute_method
        self.use_mask_features = use_mask_features
        self.trial_intercept = trial_intercept
        self.window_intercept = window_intercept
        self.max_trial_missing_rate = 0.80 if trial_intercept else 1.0
        self.max_missing_rate = 0.60 if window_intercept else 1.0
        self.max_continuous_frames = 60 if window_intercept else float('inf')

ABLATION_CONFIGS = [
    AblationConfig(name="Full_Preprocessing", use_j_stat=True, pre_ms=80, post_ms=160,
                   impute_method='pchip', use_mask_features=True, trial_intercept=True, window_intercept=True),
    AblationConfig(name="Linear_Interpolation", use_j_stat=True, pre_ms=80, post_ms=160,
                   impute_method='linear', use_mask_features=True, trial_intercept=True, window_intercept=True),
    AblationConfig(name="Without_Blink_Expansion", use_j_stat=True, pre_ms=0, post_ms=0,
                   impute_method='pchip', use_mask_features=True, trial_intercept=True, window_intercept=True),
    AblationConfig(name="No_Filtering", use_j_stat=True, pre_ms=80, post_ms=160,
                   impute_method='pchip', use_mask_features=True, trial_intercept=False, window_intercept=False),
    AblationConfig(name="Without_Mask_Features", use_j_stat=True, pre_ms=80, post_ms=160,
                   impute_method='pchip', use_mask_features=False, trial_intercept=True, window_intercept=True),
]

log_print(f"\n定义了 {len(ABLATION_CONFIGS)} 个消融实验配置:")
for cfg in ABLATION_CONFIGS:
    log_print(f"  - {cfg.name}")

# =========================
# 3. 核心预处理函数
# =========================
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
            while j < n and mask[j]: j += 1
            s = max(0, i - pre)
            e = min(n, j + post)
            padded[s:e] = True
            i = j
        else:
            i += 1
    return padded

def linear_blockwise_impute(df_block, base_features, limit_ffill=5, limit_bfill=5):
    out = df_block.copy()
    for col in base_features:
        arr = out[col].values.astype(float)
        arr[arr == 0] = np.nan
        s = pd.Series(arr)
        s = s.interpolate(method='linear')
        s = s.fillna(method='ffill', limit=limit_ffill)
        s = s.fillna(method='bfill', limit=limit_bfill)
        if s.isna().any():
            median_val = np.nanmedian(out[col].values)
            if np.isnan(median_val): median_val = 0.0
            s = s.fillna(median_val)
        out[col] = s.values
    return out

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
            if np.isnan(median_val): median_val = 0.0
            s = s.fillna(median_val)
        out[col] = s.values
    return out

def load_all_data(folder_path, config, is_train_data=True):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise ValueError(f"在 {folder_path} 没找到 CSV 文件")
    if is_train_data:
        pre_ms, post_ms = config.pre_ms, config.post_ms
        impute_method = config.impute_method
        trial_intercept = config.trial_intercept
        max_trial_missing_rate = config.max_trial_missing_rate
        print(f"  [训练集] pre_ms={pre_ms}, post_ms={post_ms}, impute={impute_method}")
    else:
        pre_ms, post_ms = ULTIMATE_PRE_MS, ULTIMATE_POST_MS
        impute_method = 'pchip'
        trial_intercept = True
        max_trial_missing_rate = ULTIMATE_MAX_TRIAL_MISSING_RATE
        print(f"  [测试集-锁定Ultimate标准] pre_ms={pre_ms}, post_ms={post_ms}")
    print(f"正在整合 {len(all_files)} 个文件...")
    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f, low_memory=False)
            if 'ParticipantID' in temp_df.columns:
                temp_df = temp_df.rename(columns={'ParticipantID': PID_COL}) if PID_COL not in temp_df.columns else temp_df.drop(columns=['ParticipantID'])
            if 'Trial' in temp_df.columns:
                temp_df = temp_df.rename(columns={'Trial': TRIAL_COL}) if TRIAL_COL not in temp_df.columns else temp_df.drop(columns=['Trial'])
            if temp_df.columns.duplicated().any():
                temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
            if TIME_COL in temp_df.columns: temp_df = temp_df.sort_values(TIME_COL)
            for col in BASE_FEATURES:
                if col not in temp_df.columns: temp_df[col] = np.nan
                else: temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                is_missing = temp_df[col].isna() | (temp_df[col] == 0)
                expanded_mask = expand_missing_mask(is_missing.values, pre_ms=pre_ms, post_ms=post_ms, sampling_ms=SAMPLING_MS)
                temp_df[col + "_Mask"] = expanded_mask.astype(float)
                temp_df.loc[expanded_mask, col] = np.nan
            blocks = [g for _, g in temp_df.groupby((temp_df[TRIAL_COL] != temp_df[TRIAL_COL].shift()).cumsum())]
            recon_blocks = []
            for b in blocks:
                trial_mask = b[MASK_FEATURES].values.astype(float)
                frame_missing_rate = trial_mask.mean(axis=1)
                severe_missing = (frame_missing_rate >= 0.5).astype(int)
                trial_missing_rate_val = severe_missing.mean()
                if trial_missing_rate_val > max_trial_missing_rate:
                    continue
                if impute_method == 'pchip':
                    recon = pchip_blockwise_impute(b, BASE_FEATURES)
                else:
                    recon = linear_blockwise_impute(b, BASE_FEATURES)
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
            print(f"[WARN] 文件 {f} 处理失败: {e}")
    if not df_list:
        raise ValueError("处理后没有可用的文件")
    return pd.concat(df_list, ignore_index=True)

def get_max_continuous_missing(mask_1d):
    padded = np.pad(mask_1d, (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0: return 0
    return np.max(ends - starts)

def create_dataset_from_full_data(full_df, train_pids, test_pid, config, scaler=None):
    X_train, Y_train, P_train, X_test, Y_test = [], [], [], [], []
    train_max_missing_rate = config.max_missing_rate if config.window_intercept else 1.0
    train_max_continuous_frames = config.max_continuous_frames if config.window_intercept else float('inf')
    test_max_missing_rate = config.max_missing_rate if config.window_intercept else 1.0
    test_max_continuous_frames = config.max_continuous_frames if config.window_intercept else float('inf')
    use_mask = config.use_mask_features
    grouped = full_df.groupby(PID_COL)
    for pid, group in grouped:
        label = 1.0 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0.0
        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()
        for block_id, block_data in group.groupby(block_ids):
            raw_base = block_data[BASE_FEATURES].values.astype(float)
            raw_mask = block_data[MASK_FEATURES].values.astype(float)
            win_rows = int(WINDOW_SIZE / 16.67)
            stride_rows = int(STRIDE / 16.67)
            if len(raw_base) < win_rows: continue
            raw_combined = np.concatenate([raw_base, raw_mask], axis=1)
            for start in range(0, len(raw_combined) - win_rows + 1, stride_rows):
                current_win = raw_combined[start: start + win_rows]
                current_mask = current_win[:, 7:]
                frame_missing_rate = current_mask.mean(axis=1)
                severe_missing = (frame_missing_rate >= 0.5).astype(int)
                missing_rate = severe_missing.mean()
                max_continuous = get_max_continuous_missing(severe_missing)
                if pid == test_pid:
                    max_missing_rate = test_max_missing_rate
                    max_continuous_frames = test_max_continuous_frames
                elif pid in train_pids:
                    max_missing_rate = train_max_missing_rate
                    max_continuous_frames = train_max_continuous_frames
                else:
                    continue
                if missing_rate <= max_missing_rate and max_continuous <= max_continuous_frames:
                    if pid == test_pid:
                        X_test.append(current_win)
                        Y_test.append(label)
                    elif pid in train_pids:
                        X_train.append(current_win)
                        Y_train.append(label)
                        P_train.append(pid)
    def filter_nan_windows(X_list, Y_list, P_list=None):
        if not X_list:
            return (np.array([]), np.array([]), np.array([])) if P_list is not None else (np.array([]), np.array([]))
        X_arr = np.array(X_list, dtype=np.float32)
        Y_arr = np.array(Y_list, dtype=np.float32)
        valid_mask = ~np.isnan(X_arr).any(axis=(1, 2))
        if P_list is not None:
            P_arr = np.array(P_list)
            return X_arr[valid_mask], Y_arr[valid_mask], P_arr[valid_mask]
        return X_arr[valid_mask], Y_arr[valid_mask]
    X_train_arr, Y_train_arr, P_train_arr = filter_nan_windows(X_train, Y_train, P_train)
    X_test_arr, Y_test_arr = filter_nan_windows(X_test, Y_test)
    if len(X_train_arr) == 0 or len(X_test_arr) == 0:
        return None, None, None, None, None, None
    train_base_feats_for_fit = X_train_arr[:, :, :7].reshape(-1, 7)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_base_feats_for_fit)
    N_tr, T, F_total = X_train_arr.shape
    N_te = X_test_arr.shape[0]
    X_train_base_scaled = scaler.transform(X_train_arr[:, :, :7].reshape(-1, 7)).reshape(N_tr, T, 7)
    X_test_base_scaled = scaler.transform(X_test_arr[:, :, :7].reshape(-1, 7)).reshape(N_te, T, 7)
    if use_mask:
        X_train_mask = X_train_arr[:, :, 7:]
        X_test_mask = X_test_arr[:, :, 7:]
        X_train_final = np.concatenate([X_train_base_scaled, X_train_mask], axis=2)
        X_test_final = np.concatenate([X_test_base_scaled, X_test_mask], axis=2)
    else:
        X_train_final = X_train_base_scaled
        X_test_final = X_test_base_scaled
    return X_train_final, Y_train_arr, P_train_arr, X_test_final, Y_test_arr, scaler

def extract_single_pid_data(full_df, target_pid, scaler, config):
    grouped = full_df.groupby(PID_COL)
    X_pid, Y_pid = [], []
    for pid, group in grouped:
        if pid != target_pid: continue
        label = 1.0 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0.0
        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()
        for block_id, block_data in group.groupby(block_ids):
            raw_base = block_data[BASE_FEATURES].values.astype(float)
            raw_mask = block_data[MASK_FEATURES].values.astype(float)
            win_rows = int(WINDOW_SIZE / 16.67)
            stride_rows = int(STRIDE / 16.67)
            if len(raw_base) < win_rows: continue
            raw_combined = np.concatenate([raw_base, raw_mask], axis=1)
            for start in range(0, len(raw_combined) - win_rows + 1, stride_rows):
                current_win = raw_combined[start: start + win_rows]
                current_mask = current_win[:, 7:]
                frame_missing_rate = current_mask.mean(axis=1)
                severe_missing = (frame_missing_rate >= 0.5).astype(int)
                missing_rate = severe_missing.mean()
                max_continuous = get_max_continuous_missing(severe_missing)
                if missing_rate <= config.max_missing_rate and max_continuous <= config.max_continuous_frames:
                    X_pid.append(current_win)
                    Y_pid.append(label)
    if not X_pid: return None, None
    X_pid_arr = np.array(X_pid, dtype=np.float32)
    Y_pid_arr = np.array(Y_pid, dtype=np.float32)
    valid_mask = ~np.isnan(X_pid_arr).any(axis=(1, 2))
    X_pid_arr = X_pid_arr[valid_mask]
    Y_pid_arr = Y_pid_arr[valid_mask]
    if len(X_pid_arr) == 0: return None, None
    N, T, F_total = X_pid_arr.shape
    X_pid_base_scaled = scaler.transform(X_pid_arr[:, :, :7].reshape(-1, 7)).reshape(N, T, 7)
    if config.use_mask_features:
        X_pid_mask = X_pid_arr[:, :, 7:]
        X_pid_final = np.concatenate([X_pid_base_scaled, X_pid_mask], axis=2)
    else:
        X_pid_final = X_pid_base_scaled
    return X_pid_final, Y_pid_arr

# =========================
# 4. 模型定义
# =========================
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
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_fc(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(self.dropout(context)).squeeze(-1)

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
                candidates = [p for p, l in zip(pool_pids, pool_labels) if l == cls and p in real_train_pids]
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

# =========================
# 5. 消融实验运行函数
# =========================
def run_single_ablation(config, full_df_train, full_df_test, pid_to_label):
    seed_everything(SEED)
    log_print(f"\n{'='*60}")
    log_print(f"🧪 开始实验: {config.name}")
    log_print(f"{'='*60}")
    all_pids = list(pid_to_label.keys())
    y_true_all, y_pred_all = [], []
    subject_rows = []
    scaler_amp = torch.cuda.amp.GradScaler()
    for i, test_pid in enumerate(all_pids):
        if MAX_FOLDS is not None and i >= MAX_FOLDS:
            break
        pool_pids = [p for p in all_pids if p != test_pid]
        pool_labels = [pid_to_label[p] for p in pool_pids]
        real_train_pids, val_pids = get_stratified_train_val_pids(pool_pids, pool_labels, test_size=0.15, random_state=SEED + i)
        X_tr, Y_tr, P_tr, _, _, scaler = create_dataset_from_full_data(full_df_train, real_train_pids, test_pid, config, scaler=None)
        if X_tr is None or len(X_tr) == 0:
            log_print(f"  [Fold {i+1}] 训练集为空，跳过"); continue
        _, _, _, X_te, Y_te, _ = create_dataset_from_full_data(full_df_test, real_train_pids, test_pid, config, scaler=scaler)
        if X_te is None or len(X_te) == 0:
            log_print(f"  [Fold {i+1}] 测试集为空，跳过"); continue
        X_val_list = []
        for v_pid in val_pids:
            x_v, y_v = extract_single_pid_data(full_df_train, v_pid, scaler, config)
            if x_v is not None and len(x_v) > 0:
                X_val_list.append((torch.tensor(x_v, dtype=torch.float32).to(DEVICE), int(np.round(np.mean(y_v)))))
        if len(X_val_list) == 0:
            log_print(f"  [Fold {i+1}] 验证集为空，跳过"); continue
        pid_counts = Counter(P_tr)
        class_0_pids = set(p for p, y in zip(P_tr, Y_tr) if y == 0)
        class_1_pids = set(p for p, y in zip(P_tr, Y_tr) if y == 1)
        n_class_0_pids, n_class_1_pids = max(1, len(class_0_pids)), max(1, len(class_1_pids))
        sample_weights = [1.0 / (pid_counts[p] * (n_class_1_pids if y == 1 else n_class_0_pids)) for p, y in zip(P_tr, Y_tr)]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        t_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        t_Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(DEVICE)
        t_X_te = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        train_loader = DataLoader(TensorDataset(t_X_tr, t_Y_tr), batch_size=BATCH_SIZE, sampler=sampler)
        input_dim = 14 if config.use_mask_features else 7
        model = AttentionNet(input_dim=input_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
        criterion = nn.BCEWithLogitsLoss()
        best_val_metric, best_thresh, best_model_weights = -1.0, 0.5, None
        patience, patience_counter = 12, 0
        for epoch in range(EPOCHS):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(): loss = criterion(model(bx), by)
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
                    with torch.cuda.amp.autocast(): probs = torch.sigmoid(model(t_X_v)).cpu().numpy()
                    val_probs_epoch.append(np.median(probs))
                    val_labels_epoch.append(y_v)
            if config.use_j_stat:
                current_best_j, current_best_th = -1.0, 0.5
                for th in np.arange(0.30, 0.71, 0.01):
                    preds = [1 if p > th else 0 for p in val_probs_epoch]
                    try:
                        tn, fp, fn, tp = confusion_matrix(val_labels_epoch, preds, labels=[0, 1]).ravel()
                        sens, spec = tp / (tp + fn + 1e-7), tn / (tn + fp + 1e-7)
                        j_stat = sens + spec - 1
                        if j_stat > current_best_j:
                            current_best_j, current_best_th = j_stat, th
                    except ValueError: continue
                val_metric, thresh = current_best_j, current_best_th
            else:
                val_metric = accuracy_score(val_labels_epoch, [1 if p > 0.5 else 0 for p in val_probs_epoch])
                thresh = 0.5
            if val_metric > best_val_metric:
                best_val_metric, best_thresh, best_model_weights = val_metric, thresh, copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience: break
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(): test_probs = torch.sigmoid(model(t_X_te)).cpu().numpy()
        final_score = np.median(test_probs)
        y_true = int(Y_te[0])
        y_pred = 1 if final_score > best_thresh else 0
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        subject_rows.append({
            "fold": i + 1,
            "pid": test_pid,
            "true_label": y_true,
            "pred_label": y_pred,
            "median_prob": float(final_score),
            "threshold": float(best_thresh),
            "n_test_windows": int(len(X_te)),
            "correct": int(y_true == y_pred),
        })
    if len(y_true_all) == 0:
        log_print(f"❌ 实验 {config.name} 失败"); return None
    final_acc, final_f1 = accuracy_score(y_true_all, y_pred_all), f1_score(y_true_all, y_pred_all)
    boot_accs, boot_f1s = [], []
    for _ in range(1000):
        boot_y_true, boot_y_pred = resample(y_true_all, y_pred_all, random_state=SEED+_)
        boot_accs.append(accuracy_score(boot_y_true, boot_y_pred))
        boot_f1s.append(f1_score(boot_y_true, boot_y_pred))
    acc_ci_low, acc_ci_high = np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)
    f1_ci_low, f1_ci_high = np.percentile(boot_f1s, 2.5), np.percentile(boot_f1s, 97.5)
    log_print(f"\n🎯 {config.name} 结果:")
    log_print(f"   Accuracy: {final_acc*100:.2f}% (95% CI: {acc_ci_low*100:.2f}% - {acc_ci_high*100:.2f}%)")
    log_print(f"   F1 Score: {final_f1*100:.2f}% (95% CI: {f1_ci_low*100:.2f}% - {f1_ci_high*100:.2f}%)")

    # 保存混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true_all, y_pred_all, labels=[0, 1],
        display_labels=["TD", "ASD"], cmap=plt.cm.Blues, ax=ax
    )
    ax.set_title(f"{config.name} (Acc: {final_acc*100:.1f}%)")
    cm_path = os.path.join(RESULTS_FOLDER, f"CM_{config.name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"   混淆矩阵已保存: {cm_path}")

    result = {
        'Experiment': config.name, 'Accuracy': f"{final_acc*100:.2f}%",
        'Accuracy_CI_Low': f"{acc_ci_low*100:.2f}%", 'Accuracy_CI_High': f"{acc_ci_high*100:.2f}%",
        'F1_Score': f"{final_f1*100:.2f}%", 'F1_CI_Low': f"{f1_ci_low*100:.2f}%", 'F1_CI_High': f"{f1_ci_high*100:.2f}%"
    }

    # 立即保存单个结果到CSV（双重保险）
    single_result_df = pd.DataFrame([result])
    single_csv_path = os.path.join(RESULTS_FOLDER, f"result_{config.name}.csv")
    single_result_df.to_csv(single_csv_path, index=False, encoding='utf-8-sig')
    log_print(f"   ✅ 单个结果已保存: {single_csv_path}")

    subject_csv_path = os.path.join(RESULTS_FOLDER, f"subject_predictions_{config.name}.csv")
    pd.DataFrame(subject_rows).to_csv(subject_csv_path, index=False, encoding='utf-8-sig')
    log_print(f"   ✅ 逐被试预测已保存: {subject_csv_path}")

    return result

# =========================
# 6. 主程序
# =========================
def run_all_ablations():
    all_results = []
    log_print("\n" + "="*60)
    log_print("🚀 开始 AttentionNet 固定窗口预处理消融实验（训练/验证/测试同配置）")
    log_print("="*60)
    log_print(f"结果将保存到: {RESULTS_FOLDER}\n")
    for idx, config in enumerate(ABLATION_CONFIGS):
        single_csv_path = os.path.join(RESULTS_FOLDER, f"result_{config.name}.csv")
        subject_csv_path = os.path.join(RESULTS_FOLDER, f"subject_predictions_{config.name}.csv")
        if os.path.exists(single_csv_path) and os.path.exists(subject_csv_path):
            log_print(f"\n[{idx+1}/{len(ABLATION_CONFIGS)}] 已存在结果，跳过: {config.name}")
            existing_result = pd.read_csv(single_csv_path).iloc[0].to_dict()
            all_results.append(existing_result)
            continue
        log_print(f"\n{'='*60}")
        log_print(f"[{idx+1}/{len(ABLATION_CONFIGS)}] 准备运行: {config.name}")
        log_print(f"{'='*60}")
        log_print(f"📥 加载数据（训练/验证/测试均按 {config.name} 配置处理）...")
        try:
            full_df = load_all_data(DATA_FOLDER, config, is_train_data=True)
            pid_to_label = {
                pid: (1 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0)
                for pid, group in full_df.groupby(PID_COL)
            }
            log_print(f"总被试: {len(pid_to_label)} 人")
            result = run_single_ablation(config, full_df, full_df, pid_to_label)
        except Exception as exc:
            log_print(f"❌ 实验 {config.name} 异常终止: {repr(exc)}")
            raise
        if result is not None:
            all_results.append(result)
            # 每完成一个实验就更新汇总结果（双重保险）
            temp_df = pd.DataFrame(all_results)
            temp_csv_path = os.path.join(RESULTS_FOLDER, "attentionnet_preprocessing_ablation_pipeline_fixed_temp.csv")
            temp_df.to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
            log_print(f"📋 汇总结果已更新: {temp_csv_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_print("\n" + "="*60)
    log_print("📊 消融实验汇总结果")
    log_print("="*60)
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_FOLDER, "attentionnet_preprocessing_ablation_pipeline_fixed.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log_print(f"✅ 最终结果已保存: {csv_path}\n")
    log_print(results_df.to_string())

    baseline_acc = float(results_df[results_df['Experiment'] == 'Full_Preprocessing']['Accuracy'].values[0].replace('%', ''))
    baseline_f1 = float(results_df[results_df['Experiment'] == 'Full_Preprocessing']['F1_Score'].values[0].replace('%', ''))
    log_print("\n" + "="*60)
    log_print("📉 各消融配置相对Full_Preprocessing的性能下降")
    log_print("="*60)
    for _, row in results_df.iterrows():
        if row['Experiment'] == 'Full_Preprocessing': continue
        acc, f1 = float(row['Accuracy'].replace('%', '')), float(row['F1_Score'].replace('%', ''))
        log_print(f"{row['Experiment']}: Accuracy下降 {baseline_acc - acc:.2f}% | F1下降 {baseline_f1 - f1:.2f}%")

    log_print(f"\n{'='*60}")
    log_print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"{'='*60}")
    return results_df

if __name__ == "__main__":
    results = run_all_ablations()
