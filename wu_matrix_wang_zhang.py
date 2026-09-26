# To generate high-dimensional matrix based on Wu et al., 2023 from dataset. Then put the matrix into Wang et al., 2024 and Zhang et al., 2025 for further result
"""
One runnable file. It does not modify Model/Model.py, the dataloaders, or the CSVs.

Run from anywhere:

    python3 wu_matrix_wang_zhang.py

That default is a short MIT smoke test (2 epochs, a few batteries). It checks that
the script runs. It does not reproduce a published accuracy number.

A longer MIT leave-one-batch run (still one process, not the 10-experiment loop
in main_MIT.py):

    python3 wu_matrix_wang_zhang.py --dataset MIT --held-out 2018-04-12 \\
        --epochs 200 --max-train 0 --max-val 0 --max-test 0 --stride 1 --batch-size 512

HUST, holding out filename group 10:

    python3 wu_matrix_wang_zhang.py --dataset HUST --held-out 10 --epochs 2

Draw charts from finished result folders, without training again:

    python3 wu_matrix_wang_zhang.py --plot-from results/wu_wang_zhang \\
        --figures-dir results/wu_wang_zhang/figures

What this file does
-------------------
1. Read MIT or HUST charge-feature tables from data/.
2. Drop 3-sigma outliers the same way as dataloader/dataloader.py (pandas std,
   ddof=1, every column including capacity).
3. Build a high-dimensional period matrix for each cycle with the TimesNet
   reshape from Wu et al., 2023: FFT along cycles, average amplitude over the
   16 feature channels, zero the DC bin, keep the top-k frequencies, period =
   window_length // frequency (integer division, as in the official TimesNet
   code). Each cycle uses a causal window that ends at that cycle. The left
   side is filled by repeating the first retained row, so later cycles are not
   visible. The window is zero-padded on the right until its length divides the
   period, then reshaped to [n_periods, period, channels]. Those rectangles
   have different periods, so each one is resized to a fixed height x width
   with bilinear interpolation. That resize is only so a batch can be stacked.
   It is not part of either paper.
4. Feed the matrix into two predictors whose formulas are kept as published.

Wang et al., 2024 (PINN4SOH), solution network and PDE:
    u = solution_u(matrix, x, t)
    u_t = du/dt, u_x = du/dx
    F = dynamical_F(x, t, u, u_x, u_t)
    f = u_t - F
    loss1 = 0.5 * MSE(u1, y1) + 0.5 * MSE(u2, y2)
    loss2 = 0.5 * MSE(f1, 0) + 0.5 * MSE(f2, 0)
    loss3 = ReLU((u2 - u1) * (y1 - y2)).sum()
    loss = loss1 + alpha * loss2 + beta * loss3
  The matrix is detached before the derivative, so u_t and u_x are still taken
  only with respect to the 16 features and the cycle index. dynamical_F stays
  35-dimensional. The first linear layer is wider because the flattened matrix
  is concatenated with the original 17 inputs. Hidden width 60, sine, dropout
  0.2, and the sine predictor match Solution_u in Model/Model.py.
  loss3 is a sum, so a batch size other than the authors' 512 changes how
  strongly beta acts. The checkpoint is the lowest validation MSE of u
  (solution_u only, same quantities as PINN.Test / PINN.Valid).

Zhang et al., 2025 (BatLiNet), shared head and mixture:
    x      = matrix(cycle) - matrix(first retained cycle)      # intra-cell
    x'     = the same quantity on a training reference cell     # inter-cell
    y_intra = w h_theta(x)
    y_delta = w h_phi(x - x')
    y_inter = y_delta + y_ref
    y_hat   = alpha_mix * y_intra + (1 - alpha_mix) * y_inter
  Training fits y_intra to SOH and y_delta to (SOH - SOH_ref). The mixture is
  applied only when predicting. Several reference cells are averaged as
  predictions, not as matrices, because h_phi is nonlinear. Reference cells
  are training batteries only.

SOH is capacity / 1.1 Ah and is a fraction. MAE, RMSE and MSE use that
fraction. MAPE is mean(|pred - true| / |true|) * 100 (percent of the ground
truth). Pairs with a zero target are skipped in MAPE; capacity after cleaning
is not expected to be zero. Metrics are sample-weighted over the evaluated
adjacent pairs (the first row of each pair, matching PINN.Test).

The 17-d vector (16 features + cycle index) is min-max scaled to [-1, 1]
inside each battery, as in Wang's loader. The period matrix is built from the
cleaned raw feature values before that scaling.

The split is leave-one-condition, not the PINN rule "battery id % 5 == 0".
MIT conditions are the three date folders. HUST conditions are the number
before the hyphen in the filename. Of the remaining batteries, 20% is
validation (sklearn train_test_split, random_state=420). Test labels are not
used to choose the checkpoint. This is not a controlled comparison with the
PINN baseline or with the published BatLiNet cell lists, and these CSVs are
not the raw Q-V curves used by Zhang et al.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import grad
from torch.nn.functional import interpolate

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURE_COLUMNS = [
    "voltage mean",
    "voltage std",
    "voltage kurtosis",
    "voltage skewness",
    "CC Q",
    "CC charge time",
    "voltage slope",
    "voltage entropy",
    "current mean",
    "current std",
    "current kurtosis",
    "current skewness",
    "CV Q",
    "CV charge time",
    "current slope",
    "current entropy",
]
NOMINAL_CAPACITY_AH = 1.1
N_FEATURES = len(FEATURE_COLUMNS)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def delete_3_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Same outlier rule as dataloader.dataloader.DF.delete_3_sigma."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    out = []
    for col in df.columns:
        series = df[col]
        rule = (series.mean() - 3 * series.std() > series) | (
            series.mean() + 3 * series.std() < series
        )
        out.extend(np.flatnonzero(rule.to_numpy()))
    if out:
        df = df.drop(index=sorted(set(int(i) for i in out))).reset_index(drop=True)
    return df


def minmax_to_unit(values: np.ndarray) -> np.ndarray:
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = high - low
    span = np.where(span == 0, 1.0, span)
    return 2.0 * (values - low) / span - 1.0


class Battery:
    def __init__(self, battery_id: str, condition: str, path: str, frame: pd.DataFrame):
        features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        capacity = frame["capacity"].to_numpy(dtype=np.float64)
        cycle = np.arange(len(frame), dtype=np.float64)
        xt = minmax_to_unit(np.column_stack([features, cycle]))
        self.battery_id = battery_id
        self.condition = condition
        self.path = path
        self.features = features.astype(np.float32)
        self.xt = xt.astype(np.float32)
        self.soh = (capacity / NOMINAL_CAPACITY_AH).astype(np.float32)
        self.matrix_cache: dict[int, np.ndarray] = {}

    @property
    def n_cycles(self) -> int:
        return int(self.soh.shape[0])


def load_battery(path: str, battery_id: str, condition: str) -> Battery | None:
    frame = pd.read_csv(path)
    missing = [name for name in FEATURE_COLUMNS + ["capacity"] if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    frame = delete_3_sigma(frame.loc[:, FEATURE_COLUMNS + ["capacity"]].copy())
    if len(frame) < 2:
        return None
    return Battery(battery_id, condition, path, frame)


def list_batteries(dataset: str) -> list[Battery]:
    batteries = []
    if dataset == "MIT":
        root = os.path.join(ROOT, "data", "MIT data")
        for condition in sorted(os.listdir(root)):
            folder = os.path.join(root, condition)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.endswith(".csv"):
                    continue
                battery_id = name[:-4]
                battery = load_battery(os.path.join(folder, name), battery_id, condition)
                if battery is not None:
                    batteries.append(battery)
    elif dataset == "HUST":
        root = os.path.join(ROOT, "data", "HUST data")
        names = [name for name in os.listdir(root) if name.endswith(".csv")]

        def sort_key(name: str) -> tuple[int, int]:
            group, cell = name[:-4].split("-")
            return int(group), int(cell)

        for name in sorted(names, key=sort_key):
            battery_id = name[:-4]
            condition = battery_id.split("-")[0]
            battery = load_battery(os.path.join(root, name), battery_id, condition)
            if battery is not None:
                batteries.append(battery)
    else:
        raise ValueError(dataset)
    if not batteries:
        raise RuntimeError(f"No batteries loaded for {dataset} under {ROOT}")
    return batteries


def dominant_periods(window: np.ndarray, k: int) -> list[int]:
    """Top-k periods. window is [T, C]. period = T // frequency, DC removed."""
    length = int(window.shape[0])
    if length <= 1:
        return [1]
    amplitude = np.abs(np.fft.rfft(window, axis=0)).mean(axis=1)
    amplitude[0] = 0.0
    k_use = min(k, amplitude.shape[0] - 1)
    chosen = np.argpartition(amplitude, -k_use)[-k_use:]
    chosen = chosen[np.argsort(amplitude[chosen])[::-1]]
    periods = []
    for frequency in chosen:
        frequency = max(int(frequency), 1)
        periods.append(int(np.clip(length // frequency, 1, length)))
    return periods


def causal_window(features: np.ndarray, cycle: int, length: int) -> np.ndarray:
    cycle = int(np.clip(cycle, 0, features.shape[0] - 1))
    start = cycle - length + 1
    if start >= 0:
        return features[start : cycle + 1]
    pad = np.repeat(features[:1], -start, axis=0)
    return np.concatenate([pad, features[: cycle + 1]], axis=0)


def matrices_from_window(window: np.ndarray, k: int, height: int, width: int) -> np.ndarray:
    """Return [k, channels, height, width] float32."""
    series = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
    channels = int(series.shape[1])
    mats = []
    for period in dominant_periods(window, k):
        pad = (period - series.shape[0] % period) % period
        padded = series
        if pad:
            padded = torch.cat([series, series.new_zeros(pad, channels)], dim=0)
        n_periods = padded.shape[0] // period
        grid = padded.reshape(n_periods, period, channels).permute(2, 0, 1).unsqueeze(0)
        resized = interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)
        mats.append(resized.squeeze(0))
    while len(mats) < k:
        mats.append(mats[-1].clone())
    return torch.stack(mats[:k], dim=0).contiguous().numpy()


def matrix_at(battery: Battery, cycle: int, args) -> np.ndarray:
    cycle = int(np.clip(cycle, 0, battery.n_cycles - 1))
    cached = battery.matrix_cache.get(cycle)
    if cached is None:
        window = causal_window(battery.features, cycle, args.window)
        cached = matrices_from_window(window, args.k, args.height, args.width)
        battery.matrix_cache[cycle] = cached
    return cached


def intra_matrix(battery: Battery, cycle: int, args) -> np.ndarray:
    return matrix_at(battery, cycle, args) - matrix_at(battery, 0, args)


class PairTable:
    def __init__(self, batteries: list[Battery], args):
        xt1, xt2, y1, y2 = [], [], [], []
        mat1, mat2, intra1, intra2 = [], [], [], []
        cycles = []
        owners = []
        for battery in batteries:
            for cycle in range(0, battery.n_cycles - 1, args.stride):
                xt1.append(battery.xt[cycle])
                xt2.append(battery.xt[cycle + 1])
                y1.append(battery.soh[cycle])
                y2.append(battery.soh[cycle + 1])
                mat1.append(matrix_at(battery, cycle, args))
                mat2.append(matrix_at(battery, cycle + 1, args))
                intra1.append(intra_matrix(battery, cycle, args))
                intra2.append(intra_matrix(battery, cycle + 1, args))
                cycles.append(cycle)
                owners.append(battery.battery_id)
        if not cycles:
            raise RuntimeError("No adjacent pairs were built. Check the battery cap and stride.")
        self.xt1 = torch.from_numpy(np.stack(xt1))
        self.xt2 = torch.from_numpy(np.stack(xt2))
        self.y1 = torch.from_numpy(np.asarray(y1, dtype=np.float32)).view(-1, 1)
        self.y2 = torch.from_numpy(np.asarray(y2, dtype=np.float32)).view(-1, 1)
        self.mat1 = torch.from_numpy(np.stack(mat1))
        self.mat2 = torch.from_numpy(np.stack(mat2))
        self.intra1 = torch.from_numpy(np.stack(intra1))
        self.intra2 = torch.from_numpy(np.stack(intra2))
        self.cycles = np.asarray(cycles, dtype=np.int64)
        self.owners = owners
        self.by_id = {battery.battery_id: battery for battery in batteries}

    def __len__(self) -> int:
        return int(self.y1.shape[0])


def cap_batteries(batteries: list[Battery], limit: int) -> list[Battery]:
    if limit <= 0 or limit >= len(batteries):
        return list(batteries)
    grouped: dict[str, list[Battery]] = {}
    for battery in batteries:
        grouped.setdefault(battery.condition, []).append(battery)
    for condition in grouped:
        grouped[condition] = sorted(grouped[condition], key=lambda item: item.battery_id)
    conditions = sorted(grouped)
    picked: list[Battery] = []
    row = 0
    while len(picked) < limit:
        grew = False
        for condition in conditions:
            if row < len(grouped[condition]):
                picked.append(grouped[condition][row])
                grew = True
                if len(picked) >= limit:
                    break
        if not grew:
            break
        row += 1
    return picked


def split_batteries(batteries: list[Battery], args) -> tuple[list[Battery], list[Battery], list[Battery]]:
    conditions = sorted({battery.condition for battery in batteries})
    if args.held_out not in conditions:
        raise SystemExit(
            f"held-out condition {args.held_out!r} is not in {dataset_label(args)}. "
            f"Available conditions: {conditions}"
        )
    test = [battery for battery in batteries if battery.condition == args.held_out]
    rest = [battery for battery in batteries if battery.condition != args.held_out]
    if len(rest) < 2:
        raise SystemExit("Need at least two batteries outside the held-out condition.")
    train, valid = train_test_split(rest, test_size=0.2, random_state=args.seed, shuffle=True)
    train = cap_batteries(list(train), args.max_train)
    valid = cap_batteries(list(valid), args.max_val)
    test = cap_batteries(test, args.max_test)
    if not train or not valid or not test:
        raise SystemExit("A split is empty after the battery cap. Raise the cap or check the held-out condition.")
    return train, valid, test


def dataset_label(args) -> str:
    return args.dataset


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP(nn.Module):
    """Same layer pattern as Model.Model.MLP, including the authors' dropout spelling."""

    def __init__(self, input_dim: int, output_dim: int, layers_num: int, hidden_dim: int, droupout: float):
        super().__init__()
        if layers_num < 2:
            raise ValueError("layers must be greater than 2")
        layers: list[nn.Module] = []
        for i in range(layers_num):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), Sin()])
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), Sin(), nn.Dropout(p=droupout)])
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 32),
            Sin(),
            nn.Linear(32, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SolutionU(nn.Module):
    def __init__(self, matrix_dim: int):
        super().__init__()
        self.encoder = MLP(
            input_dim=matrix_dim + 17,
            output_dim=32,
            layers_num=3,
            hidden_dim=60,
            droupout=0.2,
        )
        self.predictor = Predictor(32)

    def forward(self, matrix_and_xt: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.encoder(matrix_and_xt))


class MatrixEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        batch, k, channels, height, width = matrix.shape
        flat = matrix.reshape(batch, k * channels, height, width)
        return self.net(flat).flatten(1)


class ZhangSOH(nn.Module):
    def __init__(self, k: int, channels: int):
        super().__init__()
        self.h_theta = MatrixEncoder(k * channels)
        self.h_phi = MatrixEncoder(k * channels)
        self.w = nn.Linear(32, 1)
        nn.init.xavier_normal_(self.w.weight)
        nn.init.constant_(self.w.bias, 0.0)

    def intra(self, intra: torch.Tensor) -> torch.Tensor:
        return self.w(self.h_theta(intra))

    def delta(self, intra: torch.Tensor, intra_ref: torch.Tensor) -> torch.Tensor:
        return self.w(self.h_phi(intra - intra_ref))

    def mixture(self, y_intra: torch.Tensor, y_delta: torch.Tensor, y_ref: torch.Tensor, alpha_mix: float) -> torch.Tensor:
        y_inter = y_delta + y_ref
        return alpha_mix * y_intra + (1.0 - alpha_mix) * y_inter


def wang_forward(solution_u: SolutionU, dynamical_f: MLP, xt: torch.Tensor, matrix: torch.Tensor):
    """PINN forward. The matrix does not receive d/dt or d/dx."""
    xt = xt.detach().requires_grad_(True)
    features = xt[:, :-1]
    cycle = xt[:, -1:]
    flat = matrix.detach().reshape(matrix.shape[0], -1)
    u = solution_u(torch.cat([flat, features, cycle], dim=1))
    u_t = grad(u.sum(), cycle, create_graph=True, only_inputs=True, allow_unused=True)[0]
    u_x = grad(u.sum(), features, create_graph=True, only_inputs=True, allow_unused=True)[0]
    if u_t is None or u_x is None:
        raise RuntimeError("PINN derivatives were not connected to the cycle index or the features.")
    f_value = dynamical_f(torch.cat([xt, u, u_x, u_t], dim=1))
    return u, u_t - f_value


def solution_lr(epoch: int, args) -> float:
    if args.epochs <= 1 or args.warmup_epochs >= args.epochs:
        return args.warmup_lr
    if epoch < args.warmup_epochs:
        return args.warmup_lr + (args.lr - args.warmup_lr) * epoch / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    return args.final_lr + 0.5 * (args.lr - args.final_lr) * (1.0 + np.cos(np.pi * progress))


def standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    nonzero = np.abs(y_true) > 0
    if not np.any(nonzero):
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mape_percent": mape,
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "n": int(y_true.shape[0]),
        "mape_zero_targets_skipped": int(np.size(y_true) - np.count_nonzero(nonzero)),
    }


def pick_reference(pool: list[Battery], owner: str, rng: np.random.Generator) -> Battery:
    choices = [battery for battery in pool if battery.battery_id != owner]
    if not choices:
        choices = pool
    return choices[int(rng.integers(len(choices)))]


def reference_batch(table: PairTable, indices: np.ndarray, pool: list[Battery], rng: np.random.Generator, args, device):
    intra_ref_1 = []
    intra_ref_2 = []
    y_ref_1 = []
    y_ref_2 = []
    for index in indices:
        ref = pick_reference(pool, table.owners[int(index)], rng)
        cycle = int(table.cycles[int(index)])
        nxt = min(cycle + 1, ref.n_cycles - 1)
        intra_ref_1.append(intra_matrix(ref, min(cycle, ref.n_cycles - 1), args))
        intra_ref_2.append(intra_matrix(ref, nxt, args))
        y_ref_1.append(ref.soh[min(cycle, ref.n_cycles - 1)])
        y_ref_2.append(ref.soh[nxt])
    intra_ref_1 = torch.from_numpy(np.stack(intra_ref_1)).to(device)
    intra_ref_2 = torch.from_numpy(np.stack(intra_ref_2)).to(device)
    y_ref_1 = torch.from_numpy(np.asarray(y_ref_1, dtype=np.float32)).view(-1, 1).to(device)
    y_ref_2 = torch.from_numpy(np.asarray(y_ref_2, dtype=np.float32)).view(-1, 1).to(device)
    return intra_ref_1, intra_ref_2, y_ref_1, y_ref_2


def batch_tensors(table: PairTable, indices: np.ndarray, device):
    index = torch.from_numpy(indices.astype(np.int64))
    return (
        table.xt1[index].to(device),
        table.xt2[index].to(device),
        table.y1[index].to(device),
        table.y2[index].to(device),
        table.mat1[index].to(device),
        table.mat2[index].to(device),
        table.intra1[index].to(device),
        table.intra2[index].to(device),
    )


def train_one_epoch(solution_u, dynamical_f, zhang, table, pool, args, device, rng, opts) -> dict:
    solution_u.train()
    dynamical_f.train()
    zhang.train()
    mse = nn.MSELoss()
    relu = nn.ReLU()
    totals = {"loss1": 0.0, "loss2": 0.0, "loss3": 0.0, "loss_intra": 0.0, "loss_delta": 0.0}
    weight = 0
    order = rng.permutation(len(table))
    checked_update = False
    for start in range(0, len(table), args.batch_size):
        indices = order[start : start + args.batch_size]
        xt1, xt2, y1, y2, mat1, mat2, intra1, intra2 = batch_tensors(table, indices, device)
        ref1, ref2, yref1, yref2 = reference_batch(table, indices, pool, rng, args, device)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        u1, f1 = wang_forward(solution_u, dynamical_f, xt1, mat1)
        u2, f2 = wang_forward(solution_u, dynamical_f, xt2, mat2)
        loss1 = 0.5 * mse(u1, y1) + 0.5 * mse(u2, y2)
        loss2 = 0.5 * mse(f1, torch.zeros_like(f1)) + 0.5 * mse(f2, torch.zeros_like(f2))
        loss3 = relu((u2 - u1) * (y1 - y2)).sum()
        loss_wang = loss1 + args.alpha * loss2 + args.beta * loss3
        y_intra_1 = zhang.intra(intra1)
        y_intra_2 = zhang.intra(intra2)
        y_delta_1 = zhang.delta(intra1, ref1)
        y_delta_2 = zhang.delta(intra2, ref2)
        loss_intra = 0.5 * mse(y_intra_1, y1) + 0.5 * mse(y_intra_2, y2)
        loss_delta = 0.5 * mse(y_delta_1, y1 - yref1) + 0.5 * mse(y_delta_2, y2 - yref2)
        loss_zhang = loss_intra + args.lambda_delta * loss_delta
        if not torch.isfinite(loss_wang) or not torch.isfinite(loss_zhang):
            raise RuntimeError("A training loss became non-finite.")
        before = None
        if not checked_update:
            before = solution_u.encoder.net[0].weight.detach().clone()
        loss_wang.backward()
        loss_zhang.backward()
        for opt in opts:
            opt.step()
        if before is not None:
            after = solution_u.encoder.net[0].weight.detach()
            if torch.equal(before, after):
                raise RuntimeError("The Wang solution network did not receive an optimizer update.")
            checked_update = True
        n = int(indices.shape[0])
        weight += n
        totals["loss1"] += float(loss1.detach()) * n
        totals["loss2"] += float(loss2.detach()) * n
        totals["loss3"] += float(loss3.detach()) * n
        totals["loss_intra"] += float(loss_intra.detach()) * n
        totals["loss_delta"] += float(loss_delta.detach()) * n
    return {name: value / weight for name, value in totals.items()}


@torch.no_grad()
def predict_wang(solution_u: SolutionU, xt: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    flat = matrix.reshape(matrix.shape[0], -1)
    return solution_u(torch.cat([flat, xt], dim=1))


@torch.no_grad()
def collect_predictions(solution_u, zhang, table, pool, args, device, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solution_u.eval()
    zhang.eval()
    true_parts = []
    wang_parts = []
    zhang_parts = []
    order = np.arange(len(table))
    for start in range(0, len(table), args.batch_size):
        indices = order[start : start + args.batch_size]
        xt1, _, y1, _, mat1, _, intra1, _ = batch_tensors(table, indices, device)
        wang_parts.append(predict_wang(solution_u, xt1, mat1).cpu().numpy())
        true_parts.append(y1.cpu().numpy())
        mixed = []
        for _ in range(args.n_refs):
            ref1, _, yref1, _ = reference_batch(table, indices, pool, rng, args, device)
            y_intra = zhang.intra(intra1)
            y_delta = zhang.delta(intra1, ref1)
            mixed.append(zhang.mixture(y_intra, y_delta, yref1, args.alpha_mix))
        zhang_parts.append(torch.stack(mixed, dim=0).mean(dim=0).cpu().numpy())
    return (
        np.concatenate(true_parts, axis=0),
        np.concatenate(wang_parts, axis=0),
        np.concatenate(zhang_parts, axis=0),
    )


def evaluate(solution_u, zhang, table, pool, args, device, rng) -> dict:
    y_true, wang_pred, zhang_pred = collect_predictions(solution_u, zhang, table, pool, args, device, rng)
    if not np.isfinite(wang_pred).all() or not np.isfinite(zhang_pred).all():
        raise RuntimeError("Predictions contain non-finite values.")
    return {
        "wang": standard_metrics(y_true, wang_pred),
        "zhang": standard_metrics(y_true, zhang_pred),
        "y_true": y_true,
        "wang_pred": wang_pred,
        "zhang_pred": zhang_pred,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def battery_manifest(batteries: list[Battery]) -> list[dict]:
    return [
        {
            "battery_id": battery.battery_id,
            "condition": battery.condition,
            "path": os.path.relpath(battery.path, ROOT),
            "n_cycles": battery.n_cycles,
        }
        for battery in batteries
    ]


def parse_training_log(log_path: str) -> list[dict]:
    rows = []
    if not os.path.isfile(log_path):
        return rows
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("epoch "):
                continue
            fields = {}
            head, rest = line.split(":", 1)
            fields["epoch"] = int(head.replace("epoch", "").strip())
            for part in rest.strip().split():
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                fields[key] = float(value)
            rows.append(fields)
    return rows


def _figure_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_result_figures(
    figure_dir: str,
    stem: str,
    y_true: np.ndarray,
    wang_pred: np.ndarray,
    zhang_pred: np.ndarray,
    metrics: dict,
    history: list[dict],
) -> list[str]:
    """Write PNG charts for one finished test set. SOH stays a fraction."""
    plt = _figure_pyplot()
    os.makedirs(figure_dir, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    wang_pred = np.asarray(wang_pred, dtype=np.float64).reshape(-1)
    zhang_pred = np.asarray(zhang_pred, dtype=np.float64).reshape(-1)
    index = np.arange(y_true.shape[0])
    title = (
        f"{metrics.get('dataset', stem)} held-out {metrics.get('held_out_condition', '')} "
        f"n={y_true.shape[0]}  SOH fraction"
    )
    if metrics.get("preliminary_smoke"):
        title += "  (preliminary)"
    saved = []

    def finish(fig, name: str) -> None:
        path = os.path.join(figure_dir, f"{stem}_{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for axis, pred, name, color in (
        (axes[0], wang_pred, "Wang", "#1f77b4"),
        (axes[1], zhang_pred, "Zhang", "#ff7f0e"),
    ):
        axis.scatter(y_true, pred, s=14, alpha=0.75, c=color, edgecolors="none")
        low = float(min(y_true.min(), pred.min()))
        high = float(max(y_true.max(), pred.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_xlabel("True SOH")
        axis.set_ylabel(f"{name} predicted SOH")
        axis.set_title(name)
        axis.grid(True, alpha=0.3)
    fig.suptitle(title)
    finish(fig, "true_vs_pred")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].plot(index, y_true, color="black", linewidth=1.2, label="True")
    axes[0].plot(index, wang_pred, color="#1f77b4", linewidth=1.0, label="Wang")
    axes[0].set_ylabel("SOH")
    axes[0].set_title("Wang")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(index, y_true, color="black", linewidth=1.2, label="True")
    axes[1].plot(index, zhang_pred, color="#ff7f0e", linewidth=1.0, label="Zhang")
    axes[1].set_xlabel("Test pair index")
    axes[1].set_ylabel("SOH")
    axes[1].set_title("Zhang")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    finish(fig, "soh_curve")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].plot(index, np.abs(wang_pred - y_true), color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("|error|")
    axes[0].set_title("Wang absolute error")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(index, np.abs(zhang_pred - y_true), color="#ff7f0e", linewidth=1.0)
    axes[1].set_xlabel("Test pair index")
    axes[1].set_ylabel("|error|")
    axes[1].set_title("Zhang absolute error")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    finish(fig, "abs_error")

    wang_row = metrics["test_wang"]
    zhang_row = metrics["test_zhang"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    scale_labels = ["MAE", "RMSE"]
    xpos = np.arange(len(scale_labels))
    axes[0].bar(xpos - 0.18, [wang_row["mae"], wang_row["rmse"]], width=0.36, label="Wang", color="#1f77b4")
    axes[0].bar(xpos + 0.18, [zhang_row["mae"], zhang_row["rmse"]], width=0.36, label="Zhang", color="#ff7f0e")
    axes[0].set_xticks(xpos, scale_labels)
    axes[0].set_ylabel("SOH fraction")
    axes[0].set_title("MAE and RMSE")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(
        [0, 1],
        [wang_row["mape_percent"], zhang_row["mape_percent"]],
        color=["#1f77b4", "#ff7f0e"],
    )
    axes[1].set_xticks([0, 1], ["Wang", "Zhang"])
    axes[1].set_ylabel("MAPE % of true SOH")
    axes[1].set_title("MAPE")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(title)
    finish(fig, "metrics")

    if history:
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].plot(epochs, [row["loss1"] for row in history], marker="o", label="loss1 data")
        axes[0].plot(epochs, [row["loss2"] for row in history], marker="o", label="loss2 PDE")
        axes[0].plot(epochs, [row["valid_wang_mse"] for row in history], marker="o", label="valid Wang MSE")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Wang losses")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, [row["zhang_intra"] for row in history], marker="o", label="intra")
        axes[1].plot(epochs, [row["zhang_delta"] for row in history], marker="o", label="delta")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Zhang losses")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(title)
        finish(fig, "training")
    return saved


def save_summary_figure(figure_dir: str, runs: list[dict]) -> str:
    plt = _figure_pyplot()
    os.makedirs(figure_dir, exist_ok=True)
    labels = []
    mae, rmse, mape = [], [], []
    for run in runs:
        for model, color_name in (("test_wang", "Wang"), ("test_zhang", "Zhang")):
            labels.append(f"{run['dataset']}\n{color_name}")
            row = run[model]
            mae.append(row["mae"])
            rmse.append(row["rmse"])
            mape.append(row["mape_percent"])
    xpos = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].bar(xpos - 0.18, mae, width=0.36, label="MAE", color="#1f77b4")
    axes[0].bar(xpos + 0.18, rmse, width=0.36, label="RMSE", color="#2ca02c")
    axes[0].set_xticks(xpos, labels)
    axes[0].set_ylabel("SOH fraction")
    axes[0].set_title("MAE and RMSE")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(xpos, mape, color="#ff7f0e")
    axes[1].set_xticks(xpos, labels)
    axes[1].set_ylabel("MAPE % of true SOH")
    axes[1].set_title("MAPE")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Preliminary smoke test. Not a paper reproduction.")
    path = os.path.join(figure_dir, "summary_mae_rmse_mape.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_saved_runs(source: str, figures_dir: str | None) -> list[str]:
    source = os.path.abspath(source)
    if os.path.isfile(os.path.join(source, "test_predictions.npz")):
        run_dirs = [source]
    else:
        run_dirs = []
        for name in sorted(os.listdir(source)):
            folder = os.path.join(source, name)
            if os.path.isfile(os.path.join(folder, "test_predictions.npz")):
                run_dirs.append(folder)
    if not run_dirs:
        raise SystemExit(f"No test_predictions.npz under {source}")
    if figures_dir is None:
        parent = source if len(run_dirs) > 1 else os.path.dirname(source)
        figures_dir = os.path.join(parent, "figures")
    figures_dir = os.path.abspath(figures_dir)
    saved: list[str] = []
    loaded = []
    for folder in run_dirs:
        with open(os.path.join(folder, "metrics.json"), encoding="utf-8") as handle:
            metrics = json.load(handle)
        pack = np.load(os.path.join(folder, "test_predictions.npz"))
        stem = f"{metrics['dataset']}_{metrics['held_out_condition']}"
        history = parse_training_log(os.path.join(folder, "log.txt"))
        saved.extend(
            save_result_figures(
                figures_dir,
                stem,
                pack["y_true"],
                pack["wang_pred"],
                pack["zhang_pred"],
                metrics,
                history,
            )
        )
        loaded.append(metrics)
        print(f"figures for {stem} -> {figures_dir}", flush=True)
    if len(loaded) > 1:
        summary = save_summary_figure(figures_dir, loaded)
        saved.append(summary)
        print(f"summary -> {summary}", flush=True)
    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Wu et al. 2023 period matrices from MIT or HUST, then train "
            "Wang et al. 2024 and Zhang et al. 2025 on those matrices."
        )
    )
    parser.add_argument("--dataset", choices=["MIT", "HUST"], default="MIT")
    parser.add_argument("--held-out", default=None, help="MIT date folder, or HUST group number such as 10.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=3, help="0 keeps every training battery.")
    parser.add_argument("--max-val", type=int, default=1, help="0 keeps every validation battery.")
    parser.add_argument("--max-test", type=int, default=1, help="0 keeps every test battery.")
    parser.add_argument("--window", type=int, default=16, help="Causal cycle window used to form each matrix.")
    parser.add_argument("--stride", type=int, default=5, help="Subsample adjacent pairs. 1 keeps every adjacent pair.")
    parser.add_argument("--k", type=int, default=2, help="How many dominant periods to keep.")
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--n-refs", type=int, default=2, help="Reference cells averaged at validation and test.")
    parser.add_argument("--alpha", type=float, default=None, help="Wang PDE loss weight. Default 0.5 MIT, 0.6 HUST.")
    parser.add_argument("--beta", type=float, default=None, help="Wang physics loss weight. Default 0.01 MIT, 0.1 HUST.")
    parser.add_argument("--lr-F", type=float, default=None, help="Adam lr for dynamical_F. Default 1e-3 MIT, 5e-4 HUST.")
    parser.add_argument("--alpha-mix", type=float, default=0.5, help="Zhang mixture weight on the intra prediction.")
    parser.add_argument("--lambda-delta", type=float, default=1.0, help="Weight on Zhang's difference loss.")
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--warmup-lr", type=float, default=2e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--final-lr", type=float, default=2e-4)
    parser.add_argument("--zhang-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--plot-from",
        default=None,
        help="Result folder, or a parent of result folders. Draws charts and does not train.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Folder for PNG charts. With --plot-from, the default is <source>/figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plot_from:
        plot_saved_runs(args.plot_from, args.figures_dir)
        return
    if args.held_out is None:
        args.held_out = "2018-04-12" if args.dataset == "MIT" else "10"
    if args.alpha is None:
        args.alpha = 0.5 if args.dataset == "MIT" else 0.6
    if args.beta is None:
        args.beta = 0.01 if args.dataset == "MIT" else 0.1
    if args.lr_F is None:
        args.lr_F = 1e-3 if args.dataset == "MIT" else 5e-4
    if args.epochs < 1 or args.window < 1 or args.stride < 1 or args.k < 1 or args.n_refs < 1:
        raise SystemExit("epochs, window, stride, k and n-refs must be positive.")
    if args.batch_size < 1 or args.height < 1 or args.width < 1:
        raise SystemExit("batch-size, height and width must be positive.")

    set_seed(args.seed)
    device = torch.device(args.device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or os.path.join(
        ROOT, "results", "smoke_tests", "wu_wang_zhang", f"{args.dataset}_{args.held_out}_{stamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    def emit(message: str) -> None:
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    emit(f"device={device} dataset={args.dataset} held_out={args.held_out}")
    emit("Loading CSVs and applying the 3-sigma rule before the FFT.")
    batteries = list_batteries(args.dataset)
    train_b, valid_b, test_b = split_batteries(batteries, args)
    emit(
        "split sizes after cap: "
        f"train {len(train_b)}, val {len(valid_b)}, test {len(test_b)} batteries"
    )
    emit("Generating Wu period matrices for the selected adjacent pairs.")
    train_table = PairTable(train_b, args)
    valid_table = PairTable(valid_b, args)
    test_table = PairTable(test_b, args)
    example = train_b[0]
    example_cycle = min(args.window, example.n_cycles - 1)
    example_periods = dominant_periods(causal_window(example.features, example_cycle, args.window), args.k)
    emit(
        f"example {example.battery_id}: cleaned cycles={example.n_cycles}, "
        f"window={args.window}, periods at cycle {example_cycle}={example_periods}, "
        f"matrix shape={[args.k, N_FEATURES, args.height, args.width]}"
    )
    emit(
        f"pairs: train {len(train_table)}, val {len(valid_table)}, test {len(test_table)}. "
        "Each pair is two adjacent cleaned rows."
    )

    matrix_dim = args.k * N_FEATURES * args.height * args.width
    solution_u = SolutionU(matrix_dim).to(device)
    dynamical_f = MLP(input_dim=35, output_dim=1, layers_num=3, hidden_dim=60, droupout=0.2).to(device)
    zhang = ZhangSOH(args.k, N_FEATURES).to(device)
    opt_u = torch.optim.Adam(solution_u.parameters(), lr=args.warmup_lr)
    opt_f = torch.optim.Adam(dynamical_f.parameters(), lr=args.lr_F)
    opt_z = torch.optim.Adam(zhang.parameters(), lr=args.zhang_lr)
    rng = np.random.default_rng(args.seed)

    config = {
        "comment": (
            "To generate high-dimensional matrix based on Wu et al., 2023 from dataset. "
            "Then put the matrix into Wang et al., 2024 and Zhang et al., 2025 for further result"
        ),
        "commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "args": vars(args),
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "soh": "fraction = capacity / 1.1 Ah",
        "features": FEATURE_COLUMNS,
        "matrix": "cleaned raw features, causal window, top-k periods, bilinear resize",
        "xt": "per-battery min-max of 16 features plus cycle index to [-1, 1]",
        "checkpoint_rule": "lowest validation MSE of Wang solution_u",
        "train_batteries": battery_manifest(train_b),
        "valid_batteries": battery_manifest(valid_b),
        "test_batteries": battery_manifest(test_b),
        "n_pairs": {"train": len(train_table), "valid": len(valid_table), "test": len(test_table)},
    }
    write_json(os.path.join(save_dir, "config.json"), config)

    best_mse = float("inf")
    best_epoch = -1
    stale = 0
    last_epoch = -1
    history = []
    ckpt_path = os.path.join(save_dir, "best_wang_valid.pt")
    for epoch in range(args.epochs):
        lr_u = solution_lr(epoch, args)
        for group in opt_u.param_groups:
            group["lr"] = lr_u
        losses = train_one_epoch(
            solution_u, dynamical_f, zhang, train_table, train_b, args, device, rng, (opt_u, opt_f, opt_z)
        )
        valid = evaluate(solution_u, zhang, valid_table, train_b, args, device, rng)
        valid_mse = valid["wang"]["mse"]
        last_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "loss1": losses["loss1"],
                "loss2": losses["loss2"],
                "loss3": losses["loss3"],
                "zhang_intra": losses["loss_intra"],
                "zhang_delta": losses["loss_delta"],
                "valid_wang_mse": valid_mse,
            }
        )
        emit(
            f"epoch {epoch}: lr_u={lr_u:.6g} loss1={losses['loss1']:.6g} "
            f"loss2={losses['loss2']:.6g} loss3={losses['loss3']:.6g} "
            f"zhang_intra={losses['loss_intra']:.6g} zhang_delta={losses['loss_delta']:.6g} "
            f"valid_wang_mse={valid_mse:.6g}"
        )
        if valid_mse < best_mse:
            best_mse = valid_mse
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "valid_wang_mse": valid_mse,
                    "solution_u": solution_u.state_dict(),
                    "dynamical_F": dynamical_f.state_dict(),
                    "zhang": zhang.state_dict(),
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale > args.early_stop:
                emit(f"early stop at epoch {epoch}; best epoch {best_epoch}")
                break

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    solution_u.load_state_dict(checkpoint["solution_u"])
    dynamical_f.load_state_dict(checkpoint["dynamical_F"])
    zhang.load_state_dict(checkpoint["zhang"])
    test = evaluate(solution_u, zhang, test_table, train_b, args, device, np.random.default_rng(args.seed))
    preliminary = args.epochs <= 2 or args.max_train > 0 or args.max_test > 0
    report = {
        "dataset": args.dataset,
        "held_out_condition": args.held_out,
        "soh_scale": "fraction (capacity / 1.1 Ah)",
        "mape": "percent of ground truth; zero targets would be skipped",
        "aggregation": "sample-weighted over test adjacent pairs, using the first row of each pair",
        "preliminary_smoke": preliminary,
        "best_epoch": best_epoch,
        "last_epoch_trained": last_epoch,
        "checkpoint": "best validation MSE of Wang solution_u; Zhang uses that same epoch",
        "best_valid_wang_mse": best_mse,
        "test_wang": test["wang"],
        "test_zhang": test["zhang"],
        "not_a_comparison_with": [
            "PINN battery-id % 5 split",
            "published PINN4SOH numbers",
            "published BatLiNet numbers",
        ],
    }
    write_json(os.path.join(save_dir, "metrics.json"), report)
    np.savez(
        os.path.join(save_dir, "test_predictions.npz"),
        y_true=test["y_true"],
        wang_pred=test["wang_pred"],
        zhang_pred=test["zhang_pred"],
    )
    emit(f"saved {save_dir}")
    emit(
        "SOH is a fraction. MAPE is a percent of the ground-truth SOH. "
        "Metrics are sample-weighted on the test pairs."
    )
    for name in ("wang", "zhang"):
        row = test[name]
        emit(
            f"test {name}: n={row['n']} MAE={row['mae']:.6f} "
            f"MAPE%={row['mape_percent']:.4f} RMSE={row['rmse']:.6f} MSE={row['mse']:.6e}"
        )
    emit(
        f"checkpoint epoch {best_epoch} (best validation Wang MSE={best_mse:.6e}); "
        f"last trained epoch {last_epoch}."
    )
    if preliminary:
        emit("These numbers are a preliminary smoke test, not a paper reproduction.")
    figure_dir = args.figures_dir or os.path.join(save_dir, "figures")
    figure_paths = save_result_figures(
        figure_dir,
        f"{args.dataset}_{args.held_out}",
        test["y_true"],
        test["wang_pred"],
        test["zhang_pred"],
        report,
        history,
    )
    for path in figure_paths:
        emit(f"figure {path}")


if __name__ == "__main__":
    main()
