import os
from common_paths import DATA_DIR as DEFAULT_DATA_DIR
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  
import warnings
import copy
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.interpolate import PchipInterpolator
from collections import Counter

warnings.filterwarnings("ignore")

# =========================
# 0. 环境与 GPU 设置
# =========================
print("="*40)
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print("="*40)

# =========================
# 1. 全局配置
# =========================
DATA_FOLDER = str(DEFAULT_DATA_DIR)
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "model_comparison_nested_window"))
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZE = 1000    
STRIDE = 500         
WINDOW_CANDIDATES = [(2000, 1000), (1500, 750), (1000, 500), (2000, 500)]
BATCH_SIZE = 32       
LR = 0.001            
EPOCHS = 50

# 双层物理防线：临床容忍底线设置 (均采用温和比例判定)
MAX_TRIAL_MISSING_RATE = 0.80 # [宏观第一层] Trial级：单次刺激任务整体"严重缺失"率上限 (80%)
MAX_MISSING_RATE = 0.60       # [微观第二层] Window级：单窗口内"严重缺失帧"比例上限 (60%)
MAX_CONTINUOUS_FRAMES = 60    # [微观第二层] Window级：最大连续"严重缺失帧"数 (60帧约等于1000ms)

# De-blink 参数设置 (基于 60Hz 采样率, 1帧≈16.67ms)
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
# 2. 核心插值与预处理函数 (包含 Trial 级拦截与 Expanded Mask)
# =========================
def check_mask_distribution(full_df):
    print("\n" + "="*50)
    print("诊断：Mask 缺失率分布 (已同步 Expanded Mask)")
    print("="*50)
    col_mask = "Pupil Diameter Right [mm]_Mask"
    if col_mask not in full_df.columns: return
    grouped = full_df.groupby(Y_COL)
    for name, group in grouped:
        total_rows = len(group)
        missing_rows = group[col_mask].sum()
        missing_rate = (missing_rows / total_rows) * 100 if total_rows > 0 else 0
        print(f"类别 {name}: 总样本 {total_rows} 行, 缺失 {int(missing_rows)} 行 ({missing_rate:.2f}%)")
    print("="*50 + "\n")

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

def load_all_data(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "labeled_*.csv"))
    if not all_files: raise ValueError(f"在 {folder_path} 没找到 CSV 文件")
    
    print(f"正在整合 {len(all_files)} 个文件 (启用 Trial 级宏观拦截 & PCHIP)...")
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
                expanded_mask = expand_missing_mask(is_missing.values, pre_ms=PRE_MS, post_ms=POST_MS, sampling_ms=SAMPLING_MS)
                
                temp_df[col + "_Mask"] = expanded_mask.astype(float)
                temp_df.loc[expanded_mask, col] = np.nan

            blocks = [g for _, g in temp_df.groupby((temp_df[TRIAL_COL] != temp_df[TRIAL_COL].shift()).cumsum())]
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
                try: temp_df[PID_COL] = temp_df[PID_COL].astype(int)
                except Exception: temp_df[PID_COL] = pd.to_numeric(temp_df[PID_COL], errors='coerce').fillna(0).astype(int)
                df_list.append(temp_df)
                
        except Exception as e: 
            print(f"[WARN] 文件 {f} 处理失败: {e}")

    if not df_list: raise ValueError("处理后没有可用的文件")
    return pd.concat(df_list, ignore_index=True)

# =========================
# 3. 稳健的切窗与数据集构建 (包含 Window 级拦截 & PID追踪)
# =========================
def filter_nan_windows_with_pid(X_list, Y_list, P_list):
    if not X_list: return np.array([]), np.array([]), np.array([])
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    P_arr = np.array(P_list)
    valid_mask = ~np.isnan(X_arr).any(axis=(1,2))
    return X_arr[valid_mask], Y_arr[valid_mask], P_arr[valid_mask]

def filter_nan_windows(X_list, Y_list):
    if not X_list: return np.array([]), np.array([])
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    valid_mask = ~np.isnan(X_arr).any(axis=(1,2))
    return X_arr[valid_mask], Y_arr[valid_mask]

def get_max_continuous_missing(mask_1d):
    """计算 0/1 一维数组中最大连续 1 的长度"""
    padded = np.pad(mask_1d, (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0: return 0
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
    """只在内层 fold 被试上计算窗口质量诊断，不使用外层测试被试。"""
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
    """窗口参数选择只看内层训练/验证被试，避免测试被试参与窗口选择。"""
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

def create_dataset_from_full_data_pure(
    full_df, train_pids, test_pid, scaler=None, window_size_ms=WINDOW_SIZE, stride_ms=STRIDE
):
    X_train, Y_train, P_train, X_test, Y_test = [], [], [], [], []
    win_rows, stride_rows = get_window_rows(window_size_ms, stride_ms)
    
    grouped = full_df.groupby(PID_COL)
    for pid, group in grouped:
        label = 1.0 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0.0
        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()
        
        for block_id, block_data in group.groupby(block_ids):
            raw_base = block_data[BASE_FEATURES].values.astype(float)
            raw_mask = block_data[MASK_FEATURES].values.astype(float)
            if len(raw_base) < win_rows: continue
            
            raw_combined = np.concatenate([raw_base, raw_mask], axis=1)
            for start in range(0, len(raw_combined) - win_rows + 1, stride_rows):
                current_win = raw_combined[start : start + win_rows]
                
                if window_passes_quality(current_win):
                    if pid == test_pid:
                        X_test.append(current_win)
                        Y_test.append(label)
                    elif pid in train_pids:
                        X_train.append(current_win)
                        Y_train.append(label)
                        P_train.append(pid)

    X_train_arr, Y_train_arr, P_train_arr = filter_nan_windows_with_pid(X_train, Y_train, P_train)
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
    X_train_mask = X_train_arr[:, :, 7:]
    X_test_base_scaled = scaler.transform(X_test_arr[:, :, :7].reshape(-1, 7)).reshape(N_te, T, 7)
    X_test_mask = X_test_arr[:, :, 7:]
    
    X_train_final = np.concatenate([X_train_base_scaled, X_train_mask], axis=2)
    X_test_final = np.concatenate([X_test_base_scaled, X_test_mask], axis=2)

    return X_train_final, Y_train_arr, P_train_arr, X_test_final, Y_test_arr, scaler

def extract_single_pid_data_pure(
    full_df, target_pid, scaler, window_size_ms=WINDOW_SIZE, stride_ms=STRIDE
):
    grouped = full_df.groupby(PID_COL)
    X_pid, Y_pid = [], []
    win_rows, stride_rows = get_window_rows(window_size_ms, stride_ms)
    for pid, group in grouped:
        if pid != target_pid: continue
        label = 1.0 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0.0
        
        block_ids = (group[TRIAL_COL] != group[TRIAL_COL].shift()).cumsum()
        for block_id, block_data in group.groupby(block_ids):
            raw_base = block_data[BASE_FEATURES].values.astype(float)
            raw_mask = block_data[MASK_FEATURES].values.astype(float)
            if len(raw_base) < win_rows: continue
            
            raw_combined = np.concatenate([raw_base, raw_mask], axis=1)
            for start in range(0, len(raw_combined) - win_rows + 1, stride_rows):
                current_win = raw_combined[start : start + win_rows]
                
                if window_passes_quality(current_win):
                    X_pid.append(current_win)
                    Y_pid.append(label)
                
    X_pid_arr, Y_pid_arr = filter_nan_windows(X_pid, Y_pid)
    if len(X_pid_arr) == 0: return None, None
    
    N, T, F_total = X_pid_arr.shape
    X_pid_base_scaled = scaler.transform(X_pid_arr[:, :, :7].reshape(-1, 7)).reshape(N, T, 7)
    X_pid_mask = X_pid_arr[:, :, 7:]
    X_pid_final = np.concatenate([X_pid_base_scaled, X_pid_mask], axis=2)
    
    return X_pid_final, Y_pid_arr

# =========================
# 4. 模型定义
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
                candidates = [p for p, l in zip(pool_pids, pool_labels) if l == cls and p in real_train_pids]
                if candidates:
                    replace_in = np.random.choice(real_train_pids)
                    chosen = candidates[0]
                    real_train_pids.remove(chosen)
                    val_pids.append(chosen)
                    break
        return real_train_pids, val_pids
    except Exception as e:
        val_count = max(1, int(round(n * 0.15)))
        val_pids = np.random.choice(pool_pids, size=val_count, replace=False).tolist()
        real_train_pids = [p for p in pool_pids if p not in val_pids]
        return real_train_pids, val_pids


# =========================
# 模型 1: PureLSTM - 仅使用单层 LSTM + 全连接层
# =========================
class PureLSTM(nn.Module):
    def __init__(self, input_dim):
        super(PureLSTM, self).__init__()
        # 单层 LSTM，hidden_size=64，batch_first=True
        self.lstm = nn.LSTM(input_dim, 64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        # 全连接层：64 -> 32 -> 1
        self.fc = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> 直接使用输入序列
        out, _ = self.lstm(x)  # out: (batch, seq_len, 64)
        # 取 LSTM 输出序列的最后一个时间步
        last_step = out[:, -1, :]  # (batch, 64)
        return self.fc(self.dropout(last_step)).squeeze(-1)


# =========================
# 模型 2: CNNLSTM - CNN特征提取 -> 单层 LSTM -> 全连接层
# =========================
class CNNLSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNNLSTM, self).__init__()
        # CNN 模块进行特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 单层 LSTM，hidden_size=64
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # 转置为 (batch, input_dim, seq_len) 以适配 Conv1d
        x = x.transpose(1, 2)
        x = self.cnn(x)  # (batch, 64, seq_len)
        # 转置回 (batch, seq_len, 64) 以适配 LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)  # out: (batch, seq_len, 64)
        # 取 LSTM 输出序列的最后一个时间步
        last_step = out[:, -1, :]  # (batch, 64)
        return self.fc(self.dropout(last_step)).squeeze(-1)


# =========================
# 5. 训练主干与裁判评估体系
# =========================
def run_model_pipeline(model_class, model_name, full_df, pid_to_label):
    """
    接受 model_class 和 model_name 参数的函数，用于训练并评估指定模型
    """
    print("\n" + "="*60)
    print(f"开始训练模型: {model_name}")
    print("="*60)
    
    all_pids = list(pid_to_label.keys())
    y_true_all, y_pred_all = [], []
    fold_records = []
    pred_save_path = os.path.join(RESULTS_DIR, f"{model_name.lower()}_subject_predictions.csv")
    scaler_amp = torch.cuda.amp.GradScaler()
    
    for i, test_pid in enumerate(all_pids):
        print(f"\n=== Fold {i+1}/{len(all_pids)} | 盲测 PID: {test_pid} ===")
        pool_pids = [p for p in all_pids if p != test_pid]
        pool_labels = [pid_to_label[p] for p in pool_pids]
            
        real_train_pids, val_pids = get_stratified_train_val_pids(pool_pids, pool_labels, test_size=0.15, random_state=SEED + i)
        
        selected_candidate, selected_score = None, None
        for window_size_ms, stride_ms in WINDOW_CANDIDATES:
            print(f"  候选窗口 -> size: {window_size_ms} ms | stride: {stride_ms} ms")

            X_tr, Y_tr, P_tr, X_te, Y_te, scaler = create_dataset_from_full_data_pure(
                full_df, real_train_pids, test_pid,
                window_size_ms=window_size_ms, stride_ms=stride_ms
            )
            if X_tr is None or len(X_tr) == 0:
                print("    跳过候选：训练/测试窗口为空。")
                continue

            X_val_list = []
            for v_pid in val_pids:
                x_v, y_v = extract_single_pid_data_pure(
                    full_df, v_pid, scaler,
                    window_size_ms=window_size_ms, stride_ms=stride_ms
                )
                if x_v is not None and len(x_v) > 0:
                    label_val = int(np.round(np.mean(y_v)))
                    X_val_list.append((torch.tensor(x_v, dtype=torch.float32).to(DEVICE), label_val))

            if len(X_val_list) == 0:
                print("    跳过候选：验证集窗口为空。")
                continue

            # 构建基于被试的公平采样器
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
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

            print(f"    分配 -> 纯训练: {len(real_train_pids)}人(Samples:{len(X_tr)}) | 内层验证: {len(X_val_list)}人")

            t_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
            t_Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(DEVICE)

            train_loader = DataLoader(TensorDataset(t_X_tr, t_Y_tr), batch_size=BATCH_SIZE, sampler=sampler)

            model = model_class(input_dim=X_tr.shape[2]).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

            criterion = nn.BCEWithLogitsLoss()

            best_val_metric, best_val_f1, best_val_acc = -1.0, 0.0, 0.0
            best_thresh, best_model_weights = 0.5, None
            patience, patience_counter = 12, 0

            epoch_pbar = tqdm(range(EPOCHS), desc=f"Fold {i+1} {window_size_ms}/{stride_ms}", leave=False)
            for epoch in epoch_pbar:
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

                current_best_j, current_best_th = -1.0, 0.5
                current_best_preds = None
                for th in np.arange(0.30, 0.71, 0.01):
                    preds = [1 if p > th else 0 for p in val_probs_epoch]
                    tn, fp, fn, tp = confusion_matrix(val_labels_epoch, preds, labels=[0, 1]).ravel()
                    sens = tp / (tp + fn + 1e-7)
                    spec = tn / (tn + fp + 1e-7)
                    j_stat = sens + spec - 1

                    if j_stat > current_best_j:
                        current_best_j = j_stat
                        current_best_th = th
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

                if patience_counter >= patience: break

            candidate_score = (best_val_metric, best_val_f1, best_val_acc, -window_size_ms, -stride_ms)
            print(
                f"    Val J: {best_val_metric:.4f} | Val F1: {best_val_f1:.4f} | "
                f"Val Acc: {best_val_acc:.4f} | Threshold: {best_thresh:.2f}"
            )
            if best_model_weights is not None and (selected_score is None or candidate_score > selected_score):
                selected_score = candidate_score
                selected_candidate = {
                    "window_size_ms": window_size_ms,
                    "stride_ms": stride_ms,
                    "weights": best_model_weights,
                    "input_dim": X_tr.shape[2],
                    "X_te": X_te,
                    "Y_te": Y_te,
                    "best_thresh": best_thresh,
                    "val_j": best_val_metric,
                    "val_f1": best_val_f1,
                    "val_acc": best_val_acc,
                }

            del model, optimizer, scheduler, train_loader, t_X_tr, t_Y_tr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if selected_candidate is None:
            print(f"  错误：没有可用嵌套窗口候选，跳过 Fold {i+1}。")
            continue

        window_size_ms = selected_candidate["window_size_ms"]
        stride_ms = selected_candidate["stride_ms"]
        best_thresh = selected_candidate["best_thresh"]
        X_te = selected_candidate["X_te"]
        Y_te = selected_candidate["Y_te"]
        t_X_te = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

        model = model_class(input_dim=selected_candidate["input_dim"]).to(DEVICE)
        model.load_state_dict(selected_candidate["weights"])
        model.eval()
        print(
            f"  选中窗口 -> {window_size_ms}/{stride_ms} ms "
            f"(Val J={selected_candidate['val_j']:.4f}, Val F1={selected_candidate['val_f1']:.4f})"
        )
        with torch.no_grad():
            with torch.cuda.amp.autocast(): test_probs = torch.sigmoid(model(t_X_te)).cpu().numpy()
        
        final_score = np.median(test_probs)
        test_pred = 1 if final_score > best_thresh else 0
        
        y_true_all.append(int(Y_te[0]))
        y_pred_all.append(test_pred)
        fold_records.append({
            "model": model_name,
            "fold": i + 1,
            "pid": test_pid,
            "true_label": int(Y_te[0]),
            "pred": test_pred,
            "correct": int(test_pred == int(Y_te[0])),
            "window_size_ms": window_size_ms,
            "stride_ms": stride_ms,
            "inner_val_j": selected_candidate["val_j"],
            "inner_val_f1": selected_candidate["val_f1"],
            "inner_val_acc": selected_candidate["val_acc"],
            "threshold": best_thresh,
            "median_prob": float(final_score),
        })
        pd.DataFrame(fold_records).to_csv(pred_save_path, index=False)
        print(f"  优选阈值: {best_thresh:.2f} | 真: {int(Y_te[0])}, 中位数概率: {final_score:.4f}, 判决: {test_pred}")

    final_acc = accuracy_score(y_true_all, y_pred_all)
    final_f1 = f1_score(y_true_all, y_pred_all)

    print("\n" + "="*50)
    print("Bootstrap 抽样计算 95% CI...")
    boot_accs, boot_f1s = [], []
    for _ in range(1000):
        boot_y_true, boot_y_pred = resample(y_true_all, y_pred_all)
        boot_accs.append(accuracy_score(boot_y_true, boot_y_pred))
        boot_f1s.append(f1_score(boot_y_true, boot_y_pred))
    
    acc_ci_low, acc_ci_high = np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)
    f1_ci_low, f1_ci_high = np.percentile(boot_f1s, 2.5), np.percentile(boot_f1s, 97.5)
        
    print("\n" + "="*50)
    print(f"模型: {model_name}")
    print("="*50)
    print(f"Accuracy : {final_acc*100:.2f}%  (95% CI: {acc_ci_low*100:.2f}% - {acc_ci_high*100:.2f}%)")
    print(f"F1 Score : {final_f1*100:.2f}%  (95% CI: {f1_ci_low*100:.2f}% - {f1_ci_high*100:.2f}%)")
    
    # 绘制并保存混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true_all, y_pred_all, 
        display_labels=["TD", "ASD"], 
        cmap=plt.cm.Blues,
        ax=ax
    )
    ax.set_title(f"{model_name} CM (Acc: {final_acc*100:.1f}%)")
    plt.tight_layout()
    
    # 保存混淆矩阵图片
    cm_save_path = os.path.join(RESULTS_DIR, f"ConfusionMatrix_{model_name}.png")
    plt.savefig(cm_save_path, dpi=300)
    print(f"混淆矩阵已保存至: {cm_save_path}")
    plt.close(fig)

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(pred_save_path, index=False)
    print(f"被试级预测已保存至: {pred_save_path}")
    
    # 返回结果字典
    results = {
        "Model": model_name,
        "Accuracy": f"{final_acc*100:.2f}%",
        "Accuracy_95CI_Low": f"{acc_ci_low*100:.2f}%",
        "Accuracy_95CI_High": f"{acc_ci_high*100:.2f}%",
        "F1_Score": f"{final_f1*100:.2f}%",
        "F1_95CI_Low": f"{f1_ci_low*100:.2f}%",
        "F1_95CI_High": f"{f1_ci_high*100:.2f}%",
        "Subject_Predictions": pred_save_path,
    }
    
    return results


def run_ablation_experiment(selected_models=None, data_folder=DATA_FOLDER):
    """
    消融实验主函数：依次测试 PureLSTM 和 CNNLSTM 模型
    保存结果到 ablation_results.csv 和对应的混淆矩阵图片
    """
    # 加载数据
    full_df = load_all_data(data_folder)
    check_mask_distribution(full_df)
    
    pid_to_label = {}
    for pid, group in full_df.groupby(PID_COL):
        pid_to_label[pid] = 1 if str(group[Y_COL].iloc[0]).strip().upper() == "ASD" else 0
        
    all_pids = list(pid_to_label.keys())
    print(f"总被试: {len(all_pids)} 人")
    
    # 定义要测试的模型列表
    models_to_test = [
        (PureLSTM, "PureLSTM"),
        (CNNLSTM, "CNNLSTM")
    ]
    if selected_models:
        selected_models = set(selected_models)
        models_to_test = [(cls, name) for cls, name in models_to_test if name in selected_models]
    
    # 存储所有结果
    all_results = []
    
    # 循环执行每个模型的训练与测试
    for model_class, model_name in models_to_test:
        try:
            result = run_model_pipeline(model_class, model_name, full_df, pid_to_label)
            all_results.append(result)
        except Exception as e:
            print(f"模型 {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存所有结果到 CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, "architecture_comparison_results.csv")
        results_df.to_csv(csv_path, index=False)
        print("\n" + "="*60)
        print("消融实验结果汇总:")
        print("="*60)
        print(results_df.to_string(index=False))
        print(f"\n结果已保存至: {csv_path}")
    
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run LOSO architecture comparison.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["PureLSTM", "CNNLSTM"],
        default=None,
        help="Optional subset of models to run. By default both baselines are run.",
    )
    parser.add_argument(
        "--data-folder",
        default=DATA_FOLDER,
        help="Directory containing participant CSV files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_experiment(selected_models=args.models, data_folder=args.data_folder)
