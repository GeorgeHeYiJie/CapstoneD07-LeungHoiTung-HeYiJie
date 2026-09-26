"""Train and evaluate one cross-condition fold.

Checkpoint selection uses validation MSE only. Test metrics are computed
after the best validation checkpoint is restored. A smoke run is capped in
``scripts/train_cross_soh.py`` so a short epoch setting cannot loop over
the three MIT batches or the ten HUST groups.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cross_soh.data import (
    NOMINAL_CAPACITY_AH,
    SampleIndex,
    apply_scaler,
    build_windows_index,
    condition_names,
    fit_scaler,
    list_dataset,
    load_split_records,
    manifest_ids,
    split_batteries,
    stack_batch,
)
from cross_soh.metrics import regression_metrics
from cross_soh.model import CrossConditionSOH, compute_soh_losses
from cross_soh.times_matrix import series_to_period_matrices


@dataclass
class TrainConfig:
    dataset: str
    held_out: str
    save_dir: str
    data_root: str | None = None
    epochs: int = 30
    early_stop: int = 10
    batch_size: int = 32
    window: int = 32
    train_stride: int = 5
    eval_stride: int = 1
    lr: float = 1e-3
    lr_dynamics: float | None = None
    alpha: float | None = None
    beta: float | None = None
    lambda_delta: float = 1.0
    mix_alpha: float = 0.5
    d_model: int = 32
    n_layers: int = 2
    k_periods: int = 3
    n_kernels: int = 3
    n_refs: int = 4
    seed: int = 42
    split_seed: int = 420
    val_ratio: float = 0.2
    max_train_batteries: int | None = None
    max_val_batteries: int | None = None
    max_test_batteries: int | None = None
    device: str = "cpu"
    smoke: bool = False


class IndexDataset(Dataset):
    def __init__(self, samples: list[SampleIndex]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SampleIndex:
        return self.samples[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset_loss_defaults(dataset: str) -> tuple[float, float, float]:
    """Wang et al. dataset coefficients and the dynamics learning rate."""
    if dataset == "MIT":
        return 0.5, 0.01, 1e-3
    if dataset == "HUST":
        return 0.6, 0.1, 5e-4
    raise ValueError(dataset)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)
        handle.write("\n")


def _git_state() -> dict:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _environment() -> dict:
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def _choose_references(
    sample_battery_positions: list[int],
    n_train: int,
    rng: np.random.Generator,
) -> list[int]:
    if n_train < 2:
        raise ValueError("need at least two training batteries")
    chosen = []
    for position in sample_battery_positions:
        pool = [index for index in range(n_train) if index != position]
        if not pool:
            # Validation and test batteries are not inside the training list,
            # so every training index is a legal reference.
            pool = list(range(n_train))
        chosen.append(int(rng.choice(pool)))
    return chosen


def _fixed_references(n_train: int, n_refs: int) -> list[int]:
    count = min(n_refs, n_train)
    return list(range(count))


def _batch_to_torch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if key in {"battery_id", "condition", "cycle_index"}:
            continue
        out[key] = torch.from_numpy(value).to(device)
    return out


def evaluate_split(
    model: CrossConditionSOH,
    records,
    samples: list[SampleIndex],
    train_records,
    window: int,
    batch_size: int,
    n_refs: int,
    device: torch.device,
) -> dict:
    """Predict every indexed cycle. References are the first training cells."""
    model.eval()
    ref_positions = _fixed_references(len(train_records), n_refs)
    y_true = []
    y_pred = []
    battery_ids = []
    conditions = []
    cycle_index = []
    intra_pred = []
    loader = DataLoader(
        IndexDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: items,
    )
    for chunk in loader:
        accumulated = []
        intra_accumulated = []
        identity = None
        for ref_pos in ref_positions:
            batch_np = stack_batch(
                records,
                chunk,
                [ref_pos] * len(chunk),
                train_records,
                window,
            )
            identity = batch_np
            tensors = _batch_to_torch(batch_np, device)
            with torch.no_grad():
                outputs = model(tensors["window"], tensors["ref_window"], tensors["y_ref"], tensors["x0"])
            accumulated.append(outputs["pred"].cpu().numpy())
            intra_accumulated.append(outputs["pred_intra"].cpu().numpy())
        y_pred.append(np.mean(accumulated, axis=0))
        intra_pred.append(np.mean(intra_accumulated, axis=0))
        y_true.append(identity["y"])
        battery_ids.append(identity["battery_id"])
        conditions.append(identity["condition"])
        cycle_index.append(identity["cycle_index"])
    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    metrics = regression_metrics(y_true_arr, y_pred_arr)
    per_battery = []
    ids = np.concatenate(battery_ids)
    for battery_id in sorted(set(ids.tolist())):
        mask = ids == battery_id
        battery_metrics = regression_metrics(y_true_arr[mask], y_pred_arr[mask])
        battery_metrics["battery_id"] = battery_id
        per_battery.append(battery_metrics)
    macro = {
        "mae": float(np.mean([row["mae"] for row in per_battery])),
        "mape_percent": float(np.mean([row["mape_percent"] for row in per_battery])),
        "rmse": float(np.mean([row["rmse"] for row in per_battery])),
        "mse": float(np.mean([row["mse"] for row in per_battery])),
        "n_batteries": len(per_battery),
        "aggregation": "unweighted_mean_across_batteries",
    }
    return {
        "metrics": metrics,
        "per_battery_mean": macro,
        "per_battery": per_battery,
        "y_true": y_true_arr.reshape(-1),
        "y_pred": y_pred_arr.reshape(-1),
        "y_intra": np.concatenate(intra_pred).reshape(-1),
        "battery_id": ids,
        "condition": np.concatenate(conditions),
        "cycle_index": np.concatenate(cycle_index),
        "reference_policy": (
            f"mean over the first {len(ref_positions)} sorted training batteries; "
            "intra-cell term does not depend on the reference"
        ),
    }


def _run_epoch(model, optimizer, loader, train_records, window, device, cfg, rng) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "data": 0.0, "delta": 0.0, "physics": 0.0, "pde": 0.0}
    seen = 0
    for chunk in loader:
        positions = [sample.battery_pos for sample in chunk]
        refs = _choose_references(positions, len(train_records), rng)
        batch_np = stack_batch(train_records, chunk, refs, train_records, window)
        tensors = _batch_to_torch(batch_np, device)
        first = model.forward_with_residual(
            tensors["window"], tensors["ref_window"], tensors["y_ref"], tensors["x0"]
        )
        second = model.forward_with_residual(
            tensors["window_next"],
            tensors["ref_window_next"],
            tensors["y_ref_next"],
            tensors["x0"],
        )
        losses = compute_soh_losses(
            first["pred"],
            second["pred"],
            tensors["y"],
            tensors["y_next"],
            first["pred_delta"],
            second["pred_delta"],
            tensors["y_ref"],
            tensors["y_ref_next"],
            first["residual"],
            second["residual"],
            cfg.alpha,
            cfg.beta,
            cfg.lambda_delta,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        batch_n = len(chunk)
        seen += batch_n
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_n
        if not np.isfinite(totals["total"]):
            raise RuntimeError("training loss became non-finite")
    return {key: value / seen for key, value in totals.items()}


def _save_example_matrices(records, window: int, k_periods: int, path: str) -> dict:
    """Save Wu's 2D matrices for the first training battery's first window."""
    record = records[0]
    series = torch.from_numpy(record.features_norm[:window]).unsqueeze(0)
    matrices, periods, amplitudes = series_to_period_matrices(series, k_periods)
    payload = {
        "battery_id": np.array(record.battery_id),
        "periods": periods.cpu().numpy(),
        "amplitudes": amplitudes.cpu().numpy(),
    }
    for index, matrix in enumerate(matrices):
        payload[f"matrix_{index}"] = matrix.squeeze(0).cpu().numpy()
    np.savez(path, **payload)
    return {
        "battery_id": record.battery_id,
        "periods": periods.cpu().numpy().tolist(),
        "matrix_shapes": [list(matrix.squeeze(0).shape) for matrix in matrices],
        "layout": "[n_periods, period, channels]; rows are inter-period, columns are intra-period",
    }


def run_experiment(cfg: TrainConfig) -> dict:
    if cfg.window < 8:
        raise ValueError("window must be at least 8")
    if cfg.n_refs < 1:
        raise ValueError("n_refs must be positive")
    if os.path.isdir(cfg.save_dir) and os.listdir(cfg.save_dir):
        raise FileExistsError(f"{cfg.save_dir} already exists and is not empty")
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)
    alpha_default, beta_default, lr_f_default = _dataset_loss_defaults(cfg.dataset)
    if cfg.alpha is None:
        cfg.alpha = alpha_default
    if cfg.beta is None:
        cfg.beta = beta_default
    if cfg.lr_dynamics is None:
        cfg.lr_dynamics = lr_f_default

    files = list_dataset(cfg.dataset, cfg.data_root)
    split = split_batteries(
        files,
        cfg.held_out,
        val_ratio=cfg.val_ratio,
        seed=cfg.split_seed,
        max_train=cfg.max_train_batteries,
        max_val=cfg.max_val_batteries,
        max_test=cfg.max_test_batteries,
    )
    records = load_split_records(cfg.dataset, split)
    scaler = fit_scaler(records["train"])
    for group in records.values():
        apply_scaler(group, scaler)

    train_samples = build_windows_index(records["train"], cfg.train_stride, require_next=True)
    val_samples = build_windows_index(records["val"], cfg.eval_stride, require_next=False)
    test_samples = build_windows_index(records["test"], cfg.eval_stride, require_next=False)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    train_loader = DataLoader(
        IndexDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        collate_fn=lambda items: items,
    )
    device = torch.device(cfg.device)
    model = CrossConditionSOH(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        k=cfg.k_periods,
        n_kernels=cfg.n_kernels,
        mix_alpha=cfg.mix_alpha,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [p for name, p in model.named_parameters() if not name.startswith("dynamics.")], "lr": cfg.lr},
            {"params": model.dynamics.parameters(), "lr": cfg.lr_dynamics},
        ]
    )
    matrix_info = _save_example_matrices(
        records["train"],
        min(cfg.window, len(records["train"][0].features_norm)),
        cfg.k_periods,
        os.path.join(cfg.save_dir, "example_period_matrices.npz"),
    )
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    best_path = os.path.join(cfg.save_dir, "best_model.pt")
    rng = np.random.default_rng(cfg.seed)
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_losses = _run_epoch(
            model, optimizer, train_loader, records["train"], cfg.window, device, cfg, rng
        )
        validation = evaluate_split(
            model,
            records["val"],
            val_samples,
            records["train"],
            cfg.window,
            cfg.batch_size,
            cfg.n_refs,
            device,
        )
        val_mse = validation["metrics"]["mse"]
        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "config": asdict(cfg), "val_mse": val_mse},
                best_path,
            )
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train": train_losses,
                "val_mse": val_mse,
                "val_mae": validation["metrics"]["mae"],
                "improved": improved,
            }
        )
        print(
            f"epoch {epoch}: train {train_losses['total']:.6f} "
            f"data {train_losses['data']:.6f} pde {train_losses['pde']:.6f} "
            f"physics {train_losses['physics']:.6f} val_mse {val_mse:.6f}",
            flush=True,
        )
        if stale > cfg.early_stop:
            break

    torch.save({"epoch": history[-1]["epoch"], "model": model.state_dict(), "config": asdict(cfg)}, os.path.join(cfg.save_dir, "last_model.pt"))
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation = evaluate_split(
        model,
        records["val"],
        val_samples,
        records["train"],
        cfg.window,
        cfg.batch_size,
        cfg.n_refs,
        device,
    )
    test = evaluate_split(
        model,
        records["test"],
        test_samples,
        records["train"],
        cfg.window,
        cfg.batch_size,
        cfg.n_refs,
        device,
    )
    np.savez(
        os.path.join(cfg.save_dir, "test_predictions.npz"),
        y_true=test["y_true"],
        y_pred=test["y_pred"],
        y_intra=test["y_intra"],
        battery_id=test["battery_id"],
        condition=test["condition"],
        cycle_index=test["cycle_index"],
    )
    runtime = time.time() - started
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = {
        "preliminary": cfg.smoke,
        "checkpoint": "best_validation_mse",
        "best_epoch": best_epoch,
        "last_epoch": history[-1]["epoch"],
        "best_validation_mse": best_val,
        "runtime_seconds": runtime,
        "parameter_count": parameter_count,
        "nominal_capacity_Ah": NOMINAL_CAPACITY_AH[cfg.dataset],
        "test": test["metrics"],
        "test_per_battery_mean": test["per_battery_mean"],
        "test_per_battery": test["per_battery"],
        "validation": validation["metrics"],
        "validation_per_battery_mean": validation["per_battery_mean"],
        "reference_policy": test["reference_policy"],
        "period_matrix_example": matrix_info,
        "n_train_pairs": len(train_samples),
        "n_val_cycles": len(val_samples),
        "n_test_cycles": len(test_samples),
    }
    _write_json(os.path.join(cfg.save_dir, "metrics.json"), report)
    _write_json(os.path.join(cfg.save_dir, "history.json"), {"epochs": history})
    _write_json(
        os.path.join(cfg.save_dir, "config.json"),
        {
            "config": asdict(cfg),
            "loss_note": (
                "alpha and beta start from Wang et al. dataset defaults. "
                "They weight this model's residual and direction penalty, "
                "not a claim that the network matches the PINN solution."
            ),
            "environment": _environment(),
            "git": _git_state(),
            "available_conditions": condition_names(cfg.dataset),
        },
    )
    _write_json(
        os.path.join(cfg.save_dir, "split_manifest.json"),
        {
            "procedure": (
                "Leave-one-condition-out. Test batteries are the held-out MIT "
                "date batch or HUST filename group. Of the remaining batteries, "
                "20% are validation and 80% are training, split with "
                "sklearn train_test_split random_state=420. Caps keep the "
                "alphabetically first ids and are recorded separately."
            ),
            "normalization": scaler.to_dict(),
            "feature_order": [
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
                "cycle index",
            ],
            "label": "capacity / 1.1 Ah, SOH as a fraction",
            "full_train": manifest_ids(split["full_train"]),
            "full_val": manifest_ids(split["full_val"]),
            "full_test": manifest_ids(split["full_test"]),
            "used_train": manifest_ids(split["train"]),
            "used_val": manifest_ids(split["val"]),
            "used_test": manifest_ids(split["test"]),
        },
    )
    print(
        "test MAE {mae:.6f} MAPE% {mape_percent:.4f} RMSE {rmse:.6f} MSE {mse:.8f} n={n_samples}".format(
            **test["metrics"]
        ),
        flush=True,
    )
    return report
