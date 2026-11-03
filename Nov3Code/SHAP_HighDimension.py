#!/usr/bin/env python3
"""
SHAP explainability for HighDimensionMatrix-PINN_Combination.py (Wang SolutionU path).

Usage:
    python SHAP_HighDimension.py --model-path <best_wang_valid.pt> --dataset MIT
"""

import argparse
import os
import sys
import numpy as np
import torch
import shap

# ===================== 可调参数区 =====================
N_BACKGROUND = 80
N_EXPLAIN = 30
NSAMPLES = 100
WINDOW = 16
K_PERIODS = 2
HEIGHT = 8
WIDTH = 8
# ====================================================

# 确保能导入同目录下的模型文件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "hd_model",
    "HighDimensionMatrix-PINN_Combination.py"
)
hd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd)

SolutionU = hd.SolutionU
list_batteries = hd.list_batteries
matrix_at = hd.matrix_at
N_FEATURES = hd.N_FEATURES

FEATURE_NAMES = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "current mean", "current std", "current kurtosis", "current skewness",
    "CV Q", "CV charge time", "current slope", "current entropy",
    "cycle_index",
]


class ArgsProxy:
    """给 matrix_at 用的简单参数对象"""
    def __init__(self, window=16, k=2, height=8, width=8):
        self.window = window
        self.k = k
        self.height = height
        self.width = width


def build_input(battery, cycle: int, args_proxy: ArgsProxy) -> np.ndarray:
    """构造 SolutionU 的输入：展平矩阵 + xt"""
    mat = matrix_at(battery, cycle, args_proxy)          # [k, C, H, W]
    mat_flat = mat.reshape(-1).astype(np.float32)
    xt = battery.xt[cycle].astype(np.float32)            # 17-d
    return np.concatenate([mat_flat, xt])


def collect_samples(batteries, args_proxy: ArgsProxy, max_batteries=8, cycles_per_bat=30, stride=3):
    """从电池中采样输入向量"""
    samples = []
    soh_list = []
    for bat in batteries[:max_batteries]:
        n = bat.n_cycles
        for c in range(0, min(n, cycles_per_bat * stride), stride):
            if c >= n:
                break
            inp = build_input(bat, c, args_proxy)
            samples.append(inp)
            soh_list.append(bat.soh[c])
    if not samples:
        raise RuntimeError("没有采到任何样本，请检查数据路径。")
    return np.stack(samples).astype(np.float32), np.array(soh_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="SHAP for HighDimension Wang SolutionU")
    parser.add_argument("--model-path", required=True, help="best_wang_valid.pt 路径")
    parser.add_argument("--dataset", default="MIT", choices=["MIT", "HUST"])
    parser.add_argument("--device", default="cuda")
    #parser.add_argument("--n-background", type=int, default=N_BACKGROUND)
    #parser.add_argument("--n-explain", type=int, default=N_EXPLAIN)
    #parser.add_argument("--nsamples", type=int, default=NSAMPLES, help="KernelExplainer 采样次数")
    #parser.add_argument("--window", type=int, default=WINDOW)
    #parser.add_argument("--k", type=int, default=K_PERIODS)
    #parser.add_argument("--height", type=int, default=HEIGHT)
    #parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--output", default="shap_highdim.npy")
    args = parser.parse_args()
    args.n_background = N_BACKGROUND
    args.n_explain = N_EXPLAIN
    args.nsamples = NSAMPLES
    args.window = WINDOW
    args.k = K_PERIODS
    args.height = HEIGHT
    args.width = WIDTH

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 加载模型 ----------
    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

    matrix_dim = args.k * N_FEATURES * args.height * args.width
    model = SolutionU(matrix_dim).to(device)
    model.load_state_dict(ckpt["solution_u"])
    model.eval()
    print(f"SolutionU loaded (matrix_dim={matrix_dim})")

    # ---------- 准备数据 ----------
    print(f"Loading {args.dataset} batteries...")
    batteries = list_batteries(args.dataset)
    args_proxy = ArgsProxy(args.window, args.k, args.height, args.width)

    all_inputs, all_soh = collect_samples(batteries, args_proxy)
    print(f"Collected {len(all_inputs)} samples, input dim = {all_inputs.shape[1]}")

    # 随机划分背景 / 解释集
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(all_inputs), size=min(args.n_background, len(all_inputs)), replace=False)
    ex_idx = rng.choice(len(all_inputs), size=min(args.n_explain, len(all_inputs)), replace=False)
    background = all_inputs[bg_idx]
    explain_data = all_inputs[ex_idx]
    explain_soh = all_soh[ex_idx]

    # ---------- SHAP ----------
    def predict_fn(x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(x_t)
        return pred.cpu().numpy().reshape(-1)

    print(f"Background: {len(background)}, Explain: {len(explain_data)}")
    print("Running KernelExplainer (this may take a few minutes)...")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_data, nsamples=args.nsamples)

    # 统一成 2D
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    # ---------- 保存 ----------
    np.save(args.output, shap_values)
    print(f"\nSHAP values saved to {args.output}")
    print(f"Shape: {shap_values.shape}")

    # ---------- 简单报告 ----------
    mean_abs = np.abs(shap_values).mean(axis=0)
    # 只看最后 17 维（原始特征 + cycle），矩阵部分维度很高
    feature_importance = mean_abs[-17:]
    print("\n=== 全局特征重要性（平均 |SHAP|，仅 17 维原始特征）===")
    ranked = sorted(zip(FEATURE_NAMES, feature_importance), key=lambda x: -x[1])
    for name, val in ranked:
        print(f"{name:20s}: {val:.6f}")

    # 单样本示例
    print("\n=== 单样本示例（前 3 个）===")
    for i in range(min(3, len(explain_data))):
        pred = predict_fn(explain_data[i:i+1])[0]
        true = explain_soh[i]
        print(f"\n样本 {i}:")
        print(f"  真实 SOH: {true:.4f}")
        print(f"  预测 SOH: {pred:.4f}")
        print(f"  差值:     {pred - true:+.4f}")
        sv = shap_values[i, -17:]
        top = sorted(zip(FEATURE_NAMES, sv), key=lambda x: -abs(x[1]))[:5]
        print("  Top-5 SHAP:")
        for name, val in top:
            print(f"    {name}: {val:+.4f}")

    print("\n完成。可用 shap.summary_plot 进一步可视化。")


if __name__ == "__main__":
    main()