#!/usr/bin/env python3
"""
SHAP explanations for the true Wang + Zhang combined model.
Based on Molnar (2019) "Shapley Values" chapter (Model-Agnostic Methods).

Concepts used:
- Efficiency: sum of SHAP values = prediction - average prediction
- Local explanation for single battery-cycle
- Global feature importance by averaging absolute SHAP
- Monte-Carlo style approximation via KernelSHAP

Usage:
    python shap_explanation.py --data-dir ./denoised_for_matrix --model-path best_combined.pt
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import shap
from sklearn.model_selection import train_test_split

# ==================== 可调参数区 ====================
FEATURE_COLUMNS = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "current mean", "current std", "current kurtosis", "current skewness",
    "CV Q", "CV charge time", "current slope", "current entropy",
]
NOMINAL_CAPACITY = 1.1          # 可调：标称容量
N_FEATURES = 16
WINDOW = 16                     # 可调：与结合模型一致
K_PERIODS = 3                   # 可调
MATRIX_H, MATRIX_W = 8, 8       # 可调
HIDDEN = 60                     # 可调：与结合模型一致
SEED = 420
N_BACKGROUND = 50               # 可调：KernelSHAP 背景样本数（书里 Monte-Carlo 的 M）
N_EXPLAIN = 20                  # 可调：要解释的测试样本数
# ====================================================

# -------------------- 复用结合模型的关键组件（接口） --------------------
class Sin(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

class ResidualMatrixBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(K_PERIODS * N_FEATURES, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, matrix):
        b = matrix.shape[0]
        x = matrix.view(b, -1, MATRIX_H, MATRIX_W)
        return self.fc(self.conv(x).view(b, -1))

class WangPINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(17, HIDDEN), Sin(),
            torch.nn.Linear(HIDDEN, HIDDEN), Sin(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(HIDDEN, 32),
        )
        self.predictor = torch.nn.Linear(32, 1)
        self.matrix_branch = ResidualMatrixBranch()

    def solution_u(self, xt, matrix):
        u = self.predictor(self.encoder(xt))
        correction = self.matrix_branch(matrix)
        return u + 0.1 * correction

class ZhangInterCell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 32), Sin(),
            torch.nn.Linear(32, 16),
        )
        self.head_intra = torch.nn.Linear(16, 1)

    def forward(self, feat):
        h = self.encoder(feat)
        return self.head_intra(h)

class CombinedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wang = WangPINN()
        self.zhang = ZhangInterCell()
        self.mix = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, xt, matrix, feat):
        u_wang = self.wang.solution_u(xt, matrix)
        y_zhang = self.zhang(feat)
        alpha = torch.sigmoid(self.mix)
        return alpha * u_wang + (1 - alpha) * y_zhang

# -------------------- 数据与矩阵构建（与之前脚本接口一致） --------------------
def causal_window(feats, cycle, length):
    start = max(0, cycle - length + 1)
    win = feats[start:cycle + 1]
    if len(win) < length:
        pad = np.repeat(feats[:1], length - len(win), axis=0)
        win = np.concatenate([pad, win], axis=0)
    return win

def build_matrix(feats, cycle):
    win = causal_window(feats, cycle, WINDOW)
    amp = np.abs(np.fft.rfft(win, axis=0)).mean(1)
    amp[0] = 0
    idx = np.argsort(amp)[-K_PERIODS:][::-1]
    periods = [max(1, WINDOW // max(1, int(f))) for f in idx]
    mats = []
    for p in periods:
        pad = (p - len(win) % p) % p
        padded = np.pad(win, ((0, pad), (0, 0)), mode="constant")
        grid = padded.reshape(-1, p, N_FEATURES).transpose(2, 0, 1)
        t = torch.from_numpy(grid).unsqueeze(0)
        resized = torch.nn.functional.interpolate(t, size=(MATRIX_H, MATRIX_W), mode="bilinear", align_corners=False)
        mats.append(resized.squeeze(0).numpy())
    while len(mats) < K_PERIODS:
        mats.append(mats[-1])
    return np.stack(mats[:K_PERIODS])

class Battery:
    def __init__(self, path, battery_id):
        df = pd.read_csv(path)
        self.features = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.soh = (df["capacity"].values / NOMINAL_CAPACITY).astype(np.float32)
        self.n = len(self.soh)
        cycle = np.arange(self.n, dtype=np.float32)
        xt = np.column_stack([self.features, cycle])
        low, high = xt.min(0), xt.max(0)
        span = np.where(high - low == 0, 1.0, high - low)
        self.xt = (2 * (xt - low) / span - 1).astype(np.float32)
        self.id = battery_id

def load_batteries(data_dir):
    batteries = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                batteries.append(Battery(path, os.path.splitext(f)[0]))
    return batteries

# -------------------- SHAP 解释器（核心） --------------------
class ModelWrapper:
    """把 CombinedModel 包成 SHAP 能调用的函数（只接受特征矩阵）"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, X):
        # X: [N, 16] 特征
        preds = []
        with torch.no_grad():
            for i in range(len(X)):
                feat = torch.tensor(X[i:i+1], dtype=torch.float32).to(self.device)
                # 构造对应的 xt 和 matrix（这里用简化：cycle=0 作为代表，可按需扩展）
                cycle_idx = 0
                xt = torch.tensor(np.column_stack([X[i:i+1], np.array([[cycle_idx]])]), dtype=torch.float32).to(self.device)
                mat = torch.tensor(build_matrix(X[i:i+1], cycle_idx)).unsqueeze(0).to(self.device)
                pred = self.model(xt, mat, feat)
                preds.append(pred.cpu().numpy().flatten())
        return np.array(preds).flatten()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./denoised_for_matrix")
    parser.add_argument("--model-path", default="best_combined.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batteries = load_batteries(args.data_dir)
    if len(batteries) < 10:
        print("电池数量太少，请检查 data-dir")
        return

    # 准备背景数据和待解释数据
    all_feats = np.vstack([b.features for b in batteries])
    background = all_feats[np.random.choice(len(all_feats), N_BACKGROUND, replace=False)]
    explain_idx = np.random.choice(len(all_feats), N_EXPLAIN, replace=False)
    explain_data = all_feats[explain_idx]

    # 加载模型
    model = CombinedModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    wrapper = ModelWrapper(model, device)

    # KernelSHAP（对应书里 Monte-Carlo 近似 Shapley）
    explainer = shap.KernelExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_data)

    # 输出：单样本解释（书里 Efficiency 性质）
    print("\n=== 单样本 SHAP 解释（前 3 个） ===")
    for i in range(min(3, N_EXPLAIN)):
        print(f"\n样本 {i}:")
        pred = wrapper(explain_data[i:i+1])[0]
        base = explainer.expected_value
        print(f"  预测 SOH: {pred:.4f}")
        print(f"  平均预测: {base:.4f}")
        print(f"  差值: {pred - base:.4f}")
        print(f"  SHAP 之和: {shap_values[i].sum():.4f}  （应接近差值，Efficiency）")
        # 按绝对值排序显示 top 特征
        order = np.argsort(np.abs(shap_values[i]))[::-1]
        for j in order[:5]:
            print(f"    {FEATURE_COLUMNS[j]}: {shap_values[i][j]:+.4f}")

    # 全局重要性
    print("\n=== 全局特征重要性（平均 |SHAP|） ===")
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    for j in order:
        print(f"  {FEATURE_COLUMNS[j]}: {mean_abs[j]:.4f}")

    # 保存结果
    np.save("shap_values.npy", shap_values)
    np.save("explain_data.npy", explain_data)
    print("\nSHAP 值已保存为 shap_values.npy，可用 shap.summary_plot 进一步可视化。")

if __name__ == "__main__":
    main()
