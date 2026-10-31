import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.special import gamma
from scipy.interpolate import interp1d

# ==================== 可调参数区 ====================
DATA_DIR = "./your_raw_csv_folder"       # 改成你的原始 CSV 文件夹
OUTPUT_DIR = "./denoised_for_matrix"     # 输出文件夹，可直接拷到 data/MIT data/ 下
# 必须和 HighDimensionMatrix-PINN_Combination.py 里的列名一致
FEATURE_COLS = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "current mean", "current std", "current kurtosis", "current skewness",
    "CV Q", "CV charge time", "current slope", "current entropy",
]
CAPACITY_COL = "capacity"                # 容量列不降噪，原样保留

# ---- Ma et al. 2025 SG-GCL 参数 ----
SG_WINDOW = 11                           # 可调：窗口长度（奇数）
SG_POLY = 2                              # 可调：多项式阶数
GCL_SIGMA = 0.08                         # 可调：GCL 核宽度
GCL_ALPHA = 1.5                          # 可调：GCL 形状参数
GCL_ITER = 3                             # 可调：迭代次数

# ---- Huang et al. 2025 多尺度参数 ----
SCALES = [2, 4, 8, 16]                   # 可调：池化尺度
FUSION_MODE = "weighted"                 # 可调："average" 或 "weighted"
SCALE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]     # 可调：各尺度权重
# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generalized_correntropy(residual, sigma, alpha):
    """Ma et al. 广义相关熵"""
    c = (alpha ** (1 - 1.0 / alpha)) / (2.0 * gamma(1.0 / alpha))
    return c * np.exp(-(np.abs(residual) ** alpha) / (sigma ** alpha))

def sg_gcl_denoise(signal, window_length, polyorder, sigma, alpha, n_iter):
    """Ma et al. 2025 SG-GCL 降噪"""
    if len(signal) < window_length:
        return signal.copy()
    filtered = savgol_filter(signal, window_length, polyorder, mode='interp')
    for _ in range(n_iter):
        residual = signal - filtered
        w = generalized_correntropy(residual, sigma, alpha)
        w = w / (np.max(w) + 1e-8)
        filtered = filtered + 0.4 * w * residual   # 0.4 是修正强度，可调
    return filtered

def hierarchical_pool(signal, scale):
    """Huang et al. 单尺度 hierarchical pooling"""
    L = len(signal)
    if scale >= L or scale < 2:
        return signal.copy()
    n_seg = L // scale
    pooled = np.array([np.mean(signal[i*scale:(i+1)*scale]) for i in range(n_seg)])
    x_old = np.linspace(0, 1, len(pooled))
    x_new = np.linspace(0, 1, L)
    f = interp1d(x_old, pooled, kind='linear', fill_value='extrapolate')
    return f(x_new)

def multi_scale_decompose_and_fuse(signal, scales, fusion_mode, scale_weights):
    """Huang et al. 2025 多尺度分解 + 融合"""
    scale_results = []
    for s in scales:
        scale_results.append(hierarchical_pool(signal, s))
    arr = np.stack(scale_results, axis=0)
    if fusion_mode == "average":
        return np.mean(arr, axis=0)
    w = np.array(scale_weights)
    if len(w) != len(scales):
        w = np.ones(len(scales)) / len(scales)
    else:
        w = w / w.sum()
    return np.sum(arr * w[:, None], axis=0)

def process_one_file(filepath):
    """处理单个 CSV，输出列名与原文件一致（特征已降噪）"""
    df = pd.read_csv(filepath)
    
    # 检查必要列
    missing = [c for c in FEATURE_COLS + [CAPACITY_COL] if c not in df.columns]
    if missing:
        print(f"跳过 {filepath}：缺少列 {missing}")
        return
    
    df_out = df.copy()
    
    for col in FEATURE_COLS:
        sig = df[col].values.astype(float)
        
        # 第一步：Ma SG-GCL
        denoised_ma = sg_gcl_denoise(
            sig, SG_WINDOW, SG_POLY, GCL_SIGMA, GCL_ALPHA, GCL_ITER
        )
        
        # 第二步：Huang 多尺度
        fused = multi_scale_decompose_and_fuse(
            denoised_ma, SCALES, FUSION_MODE, SCALE_WEIGHTS
        )
        
        # 用降噪结果覆盖原列（capacity 不动）
        df_out[col] = fused
    
    # 写出，文件名保持原样
    basename = os.path.basename(filepath)
    out_path = os.path.join(OUTPUT_DIR, basename)
    df_out.to_csv(out_path, index=False)
    print(f"已处理：{basename} → {out_path}")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"数据文件夹不存在：{DATA_DIR}")
        return
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("没有找到 CSV 文件")
        return
    print(f"找到 {len(csv_files)} 个文件，开始处理...")
    for f in csv_files:
        process_one_file(os.path.join(DATA_DIR, f))
    print("完成。把", OUTPUT_DIR, "里的 CSV 拷到 data/MIT data/ 对应文件夹即可。")

if __name__ == "__main__":
    main()
