#!/usr/bin/env python3
"""
True combination of Wang et al. 2024 (PINN) and Zhang et al. 2025 (BatLiNet)
on denoised features. Residual dual-path design so matrix does not break physics.

Usage:
    python true_wang_zhang_combination.py --data-dir ./denoised_for_matrix --epochs 100
"""

from __future__ import annotations
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad
from sklearn.model_selection import train_test_split

# ==================== 可调参数区 ====================
FEATURE_COLUMNS = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "current mean", "current std", "current kurtosis", "current skewness",
    "CV Q", "CV charge time", "current slope", "current entropy",
]
NOMINAL_CAPACITY = 1.1          # 可调：标称容量 Ah
N_FEATURES = 16
WINDOW = 16                     # 可调：TimesNet 因果窗口长度
K_PERIODS = 3                   # 可调：保留的 top-k 周期数
MATRIX_H, MATRIX_W = 8, 8       # 可调：矩阵 resize 后的高宽
HIDDEN = 60                     # 可调：Wang 隐藏层宽度
ALPHA_PHYS = 0.1                # 可调：物理损失权重（对应 Wang loss2）
BETA_MONO = 0.01                # 可调：单调性损失权重（对应 Wang loss3）
MIX_INIT = 0.5                  # 可调：Wang/Zhang 初始混合权重
SEED = 420
# ====================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# -------------------- 数据加载（接口：读降噪后的 CSV） --------------------
class Battery:
    def __init__(self, path, battery_id, condition):
        df = pd.read_csv(path)
        self.features = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.capacity = df["capacity"].values.astype(np.float32)
        self.soh = (self.capacity / NOMINAL_CAPACITY).astype(np.float32)
        self.n = len(self.soh)
        self.id = battery_id
        self.condition = condition
        # 17 维输入（特征 + cycle index），minmax 到 [-1,1]
        cycle = np.arange(self.n, dtype=np.float32)
        xt = np.column_stack([self.features, cycle])
        low, high = xt.min(0), xt.max(0)
        span = np.where(high - low == 0, 1.0, high - low)
        self.xt = (2 * (xt - low) / span - 1).astype(np.float32)

def load_batteries(data_dir):
    batteries = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue
            path = os.path.join(root, f)
            bid = os.path.splitext(f)[0]
            cond = os.path.basename(root)
            batteries.append(Battery(path, bid, cond))
    return batteries

# -------------------- 高维矩阵（因果 TimesNet 风格） --------------------
def causal_window(feats, cycle, length):
    start = max(0, cycle - length + 1)
    win = feats[start:cycle+1]
    if len(win) < length:
        pad = np.repeat(feats[:1], length - len(win), axis=0)
        win = np.concatenate([pad, win], axis=0)
    return win

def build_matrix(feats, cycle, k, h, w):
    win = causal_window(feats, cycle, WINDOW)
    # FFT 沿 cycle 维，取 top-k 周期
    amp = np.abs(np.fft.rfft(win, axis=0)).mean(1)
    amp[0] = 0
    idx = np.argsort(amp)[-k:][::-1]
    periods = [max(1, WINDOW // max(1, int(f))) for f in idx]
    mats = []
    for p in periods:
        pad = (p - len(win) % p) % p
        padded = np.pad(win, ((0, pad), (0, 0)), mode="constant")
        grid = padded.reshape(-1, p, N_FEATURES).transpose(2, 0, 1)
        # bilinear resize
        t = torch.from_numpy(grid).unsqueeze(0)
        resized = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        mats.append(resized.squeeze(0).numpy())
    while len(mats) < k:
        mats.append(mats[-1])
    return np.stack(mats[:k])  # [k, C, H, W]

# -------------------- 模型：真正结合 --------------------
class ResidualMatrixBranch(nn.Module):
    """辅助矩阵分支：输出标量校正量，不破坏主路径物理梯度"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(K_PERIODS * N_FEATURES, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, matrix):
        # matrix: [B, k, C, H, W] -> [B, k*C, H, W]
        b = matrix.shape[0]
        x = matrix.view(b, -1, MATRIX_H, MATRIX_W)
        return self.fc(self.conv(x).view(b, -1))

class WangPINN(nn.Module):
    """Wang et al. 2024 主路径：17 维输入 + 物理约束"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(17, HIDDEN), Sin(),
            nn.Linear(HIDDEN, HIDDEN), Sin(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN, 32),
        )
        self.predictor = nn.Linear(32, 1)
        self.dyn_F = nn.Sequential(          # 对应 dynamical_F
            nn.Linear(17 + 1 + 17, 32), Sin(),
            nn.Linear(32, 1),
        )
        self.matrix_branch = ResidualMatrixBranch()

    def solution_u(self, xt, matrix):
        u = self.predictor(self.encoder(xt))
        correction = self.matrix_branch(matrix)
        return u + 0.1 * correction          # 0.1 是校正强度，可调

    def physics_residual(self, xt, u, matrix):
        # 只对主路径求导数，矩阵 detach
        u_t = grad(u.sum(), xt, create_graph=True)[0][:, -1:]
        u_x = grad(u.sum(), xt, create_graph=True)[0][:, :-1]
        # dynamical_F 输入维度保持与原论文一致
        f_input = torch.cat([xt, u.detach(), u_x, u_t], dim=1)
        F = self.dyn_F(f_input)
        return u_t - F

class ZhangInterCell(nn.Module):
    """Zhang et al. 2025 inter-cell 差异学习"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_FEATURES, 32), Sin(),
            nn.Linear(32, 16),
        )
        self.head_intra = nn.Linear(16, 1)
        self.head_delta = nn.Linear(16, 1)

    def forward(self, feat, feat_ref=None):
        h = self.encoder(feat)
        y_intra = self.head_intra(h)
        if feat_ref is None:
            return y_intra
        h_ref = self.encoder(feat_ref)
        y_delta = self.head_delta(h - h_ref)
        return y_intra, y_delta

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wang = WangPINN()
        self.zhang = ZhangInterCell()
        self.mix = nn.Parameter(torch.tensor(MIX_INIT))  # 可学习混合权重

    def forward(self, xt, matrix, feat, feat_ref=None):
        u_wang = self.wang.solution_u(xt, matrix)
        if feat_ref is None:
            y_zhang = self.zhang(feat)
        else:
            y_intra, y_delta = self.zhang(feat, feat_ref)
            y_zhang = y_intra  # 训练时用 intra；推理时再加 delta
        alpha = torch.sigmoid(self.mix)
        return alpha * u_wang + (1 - alpha) * y_zhang, u_wang

# -------------------- 训练循环 --------------------
def train(model, train_bat, val_bat, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        total_loss = 0
        for bat in train_bat:
            for c in range(0, bat.n - 1, 2):  # stride=2 可调
                xt = torch.tensor(bat.xt[c:c+2]).to(device)
                mat = torch.tensor(np.stack([
                    build_matrix(bat.features, c, K_PERIODS, MATRIX_H, MATRIX_W),
                    build_matrix(bat.features, c+1, K_PERIODS, MATRIX_H, MATRIX_W)
                ])).to(device)
                y = torch.tensor(bat.soh[c:c+2]).view(-1, 1).to(device)
                feat = torch.tensor(bat.features[c:c+2]).to(device)

                pred, u = model(xt, mat, feat)
                # Wang 物理损失
                f = model.wang.physics_residual(xt, u, mat)
                loss_data = nn.functional.mse_loss(pred, y)
                loss_phys = nn.functional.mse_loss(f, torch.zeros_like(f))
                # 单调性
                mono = torch.relu((u[1] - u[0]) * (y[0] - y[1])).sum()
                loss = loss_data + ALPHA_PHYS * loss_phys + BETA_MONO * mono
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
        # 简单验证
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bat in val_bat[:5]:
                for c in range(0, bat.n-1, 4):
                    xt = torch.tensor(bat.xt[c:c+1]).to(device)
                    mat = torch.tensor(build_matrix(bat.features, c, K_PERIODS, MATRIX_H, MATRIX_W)).unsqueeze(0).to(device)
                    feat = torch.tensor(bat.features[c:c+1]).to(device)
                    pred, _ = model(xt, mat, feat)
                    val_mae += abs(pred.item() - bat.soh[c])
        print(f"Epoch {ep}: train_loss={total_loss:.4f}, val_mae≈{val_mae:.4f}")
        if val_mae < best:
            best = val_mae
            torch.save(model.state_dict(), "best_combined.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./denoised_for_matrix")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batteries = load_batteries(args.data_dir)
    train_b, val_b = train_test_split(batteries, test_size=0.2, random_state=SEED)
    model = CombinedModel().to(device)
    train(model, train_b, val_b, args.epochs, device)
    print("训练完成，最佳模型已保存为 best_combined.pt")

if __name__ == "__main__":
    main()
