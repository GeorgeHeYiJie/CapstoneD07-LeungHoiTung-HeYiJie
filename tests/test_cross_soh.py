"""Checks for the cross-condition SOH pipeline."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_soh.data import (
    BatteryRecord,
    apply_scaler,
    build_windows_index,
    delete_3_sigma,
    fit_scaler,
    list_dataset,
    make_window,
    read_battery_table,
    split_batteries,
)
from cross_soh.metrics import regression_metrics
from cross_soh.model import CrossConditionSOH, compute_soh_losses
from cross_soh.times_matrix import TimesBlock, dominant_periods, reshape_to_2d
from cross_soh.train import TrainConfig, run_experiment


def test_fft_period_of_a_pure_sine():
    time = torch.arange(64, dtype=torch.float32)
    signal = torch.sin(2 * torch.pi * time / 8)
    series = signal.view(1, 64, 1).repeat(1, 1, 3)
    periods, amplitudes = dominant_periods(series, k=1)
    assert int(periods[0]) == 8
    assert amplitudes.shape == (1, 1)


def test_reshape_puts_one_period_in_each_row():
    series = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    matrix = reshape_to_2d(series, period=4)
    assert matrix.shape == (1, 2, 4, 1)
    assert matrix[0, :, :, 0].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_reshape_zero_pads_the_tail():
    series = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    matrix = reshape_to_2d(series, period=4)
    assert matrix.shape == (1, 2, 4, 1)
    assert matrix[0, 1, :, 0].tolist() == [4, 0, 0, 0]


def test_times_block_keeps_the_sequence_shape():
    block = TimesBlock(channels=4, k=2, n_kernels=2)
    series = torch.randn(2, 16, 4)
    assert block(series).shape == series.shape


def test_wang_direction_penalty_matches_the_published_example():
    pred_1 = torch.tensor([[0.93]])
    pred_2 = torch.tensor([[0.94]])
    y_1 = torch.tensor([[0.95]])
    y_2 = torch.tensor([[0.94]])
    zeros = torch.zeros(1, 1)
    losses = compute_soh_losses(
        pred_1, pred_2, y_1, y_2, zeros, zeros, zeros, zeros, zeros, zeros, alpha=0.0, beta=1.0, lambda_delta=0.0
    )
    assert torch.isclose(losses["physics"], torch.tensor(0.0001))


def test_standard_mape_uses_the_ground_truth_denominator():
    metrics = regression_metrics(np.array([1.0, 0.5]), np.array([1.1, 0.25]))
    # |0.1|/1 + |0.25|/0.5 = 0.1 + 0.5, mean 0.3, times 100.
    assert abs(metrics["mape_percent"] - 30.0) < 1e-8
    assert abs(metrics["mae"] - 0.175) < 1e-8


def test_shared_head_adds_the_reference_soh():
    torch.manual_seed(0)
    model = CrossConditionSOH(d_model=4, n_layers=1, k=2, n_kernels=2, mix_alpha=0.5)
    window = torch.randn(2, 8, 17)
    reference = torch.randn(2, 8, 17)
    y_ref = torch.tensor([[0.9], [0.8]])
    x0 = torch.randn(2, 17)
    outputs = model(window, reference, y_ref, x0)
    assert torch.allclose(outputs["pred_inter"], outputs["pred_delta"] + y_ref)
    fused = 0.5 * outputs["pred_intra"] + 0.5 * outputs["pred_inter"]
    assert torch.allclose(outputs["pred"], fused)
    assert model.intra_encoder is not model.inter_encoder
    assert model.head.weight.shape == (1, 4)


def test_physics_backward_reaches_both_branches_and_the_dynamics_net():
    torch.manual_seed(0)
    model = CrossConditionSOH(d_model=4, n_layers=1, k=2, n_kernels=2, dynamics_hidden=8, mix_alpha=0.5)
    model.train()
    window = torch.randn(2, 8, 17)
    reference = torch.randn(2, 8, 17)
    y_ref = torch.rand(2, 1)
    x0 = torch.randn(2, 17)
    first = model.forward_with_residual(window, reference, y_ref, x0)
    second = model.forward_with_residual(window + 0.01, reference, y_ref, x0)
    losses = compute_soh_losses(
        first["pred"],
        second["pred"],
        torch.full((2, 1), 0.95),
        torch.full((2, 1), 0.94),
        first["pred_delta"],
        second["pred_delta"],
        y_ref,
        y_ref,
        first["residual"],
        second["residual"],
        alpha=0.5,
        beta=0.01,
        lambda_delta=1.0,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"]).item()
    groups = {
        "intra": model.intra_encoder.embed.weight.grad,
        "inter": model.inter_encoder.embed.weight.grad,
        "head": model.head.weight.grad,
        "dynamics": model.dynamics.net[0].weight.grad,
    }
    for name, gradient in groups.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0, name


def test_cleaning_drops_a_3_sigma_outlier():
    # Enough inliers that the spike exceeds mean + 3 sample standard deviations.
    values = np.concatenate([np.linspace(-0.2, 0.2, 40), np.array([25.0])])
    frame = pd.DataFrame({"value": values, "capacity": np.full(len(values), 1.1)})
    assert frame["value"].mean() + 3 * frame["value"].std() < 25.0
    cleaned = delete_3_sigma(frame)
    assert 25.0 not in cleaned["value"].tolist()
    assert len(cleaned) == 40


def test_scaler_is_fit_on_training_batteries_only():
    train = [
        BatteryRecord("a", "c1", "a.csv", np.array([0.0, 1.0]), np.zeros((2, 16)), np.array([1.0, 0.9]))
    ]
    train[0].features[:, 0] = [0.0, 1.0]
    scaler = fit_scaler(train)
    test = BatteryRecord("b", "c2", "b.csv", np.array([0.0]), np.full((1, 16), 100.0), np.array([0.8]))
    apply_scaler([test], scaler)
    assert test.features_norm[0, 0] > 1.0


def test_leave_one_condition_split_is_disjoint():
    mit = list_dataset("MIT")
    split = split_batteries(mit, "2018-04-12", seed=420)
    train_ids = {item[0] for item in split["train"]}
    val_ids = {item[0] for item in split["val"]}
    test_ids = {item[0] for item in split["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert {item[1] for item in split["test"]} == {"2018-04-12"}
    assert "2018-04-12" not in {item[1] for item in split["train"] + split["val"]}
    hust = list_dataset("HUST")
    hust_split = split_batteries(hust, "10", seed=420)
    assert {item[1] for item in hust_split["test"]} == {"10"}
    assert "10" not in {item[1] for item in hust_split["train"] + hust_split["val"]}


def test_real_mit_table_uses_the_1_1_ah_divisor():
    path = "data/MIT data/2017-05-12/2017-05-12_battery-1.csv"
    cycle, features, soh = read_battery_table(path, 1.1)
    raw = pd.read_csv(path)
    assert features.shape[1] == 16
    assert len(cycle) == len(soh)
    assert np.isfinite(soh).all()
    assert soh.max() <= raw["capacity"].max() / 1.1 + 1e-6


def test_causal_window_repeats_the_first_cycle():
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    window = make_window(features, cycle_pos=1, window=4)
    assert window.shape == (4, 3)
    assert np.allclose(window[0], features[0])
    assert np.allclose(window[-1], features[1])


def _write_cell(directory, name, length=24):
    os.makedirs(directory, exist_ok=True)
    rows = []
    for index in range(length):
        features = [0.2 * np.sin(index / 3.0 + column) for column in range(16)]
        capacity = 1.1 - 0.004 * index
        rows.append(features + [capacity])
    frame = pd.DataFrame(rows, columns=[
        "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
        "CC Q", "CC charge time", "voltage slope", "voltage entropy",
        "current mean", "current std", "current kurtosis", "current skewness",
        "CV Q", "CV charge time", "current slope", "current entropy", "capacity",
    ])
    frame.to_csv(os.path.join(directory, name), index=False)


def test_one_epoch_on_synthetic_hust_cells_is_finite(tmp_path):
    root = tmp_path / "hust"
    for name in ["1-1.csv", "1-2.csv", "1-3.csv", "2-1.csv"]:
        _write_cell(root, name)
    save = tmp_path / "run"
    report = run_experiment(
        TrainConfig(
            dataset="HUST",
            held_out="2",
            save_dir=str(save),
            data_root=str(root),
            epochs=1,
            early_stop=5,
            batch_size=4,
            window=8,
            train_stride=4,
            eval_stride=4,
            d_model=4,
            n_layers=1,
            k_periods=2,
            n_kernels=2,
            n_refs=1,
            seed=42,
            device="cpu",
            smoke=True,
        )
    )
    assert np.isfinite(report["test"]["mae"])
    assert np.isfinite(report["test"]["mse"])
    assert report["checkpoint"] == "best_validation_mse"
    assert (save / "example_period_matrices.npz").exists()
    matrices = np.load(save / "example_period_matrices.npz")
    assert "matrix_0" in matrices
    assert matrices["matrix_0"].ndim == 3
    # Training samples exist and the held-out cell is the only test battery.
    manifest = __import__("json").loads((save / "split_manifest.json").read_text())
    assert [row["battery_id"] for row in manifest["used_test"]] == ["2-1"]
    assert all(row["condition"] != "2" for row in manifest["used_train"])
    samples = build_windows_index(
        [BatteryRecord("x", "1", "x", np.zeros(4), np.zeros((4, 16)), np.zeros(4))],
        stride=2,
        require_next=True,
    )
    assert [sample.cycle_pos for sample in samples] == [0, 2]
