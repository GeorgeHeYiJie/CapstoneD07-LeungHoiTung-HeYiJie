"""Standard SOH regression metrics.

Ground truth is ``y_true`` and the model output is ``y_pred``. MAPE divides
by the ground truth and is reported as a percentage. This is separate from
``utils.util.eval_metrix``, whose callers have a reversed-argument MAPE.
"""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sample-weighted MAE, MAPE (%), RMSE and MSE.

    SOH is a fraction of nominal capacity. MAE and RMSE use that fraction.
    MSE is the square of that fraction. MAPE is ``100 * mean(|e| / |y_true|)``.
    Samples with ``|y_true| < 1e-8`` stay in MAE, RMSE and MSE, and are left
    out of MAPE. The excluded count is returned as ``mape_excluded``.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("no samples to score")
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    usable = np.abs(y_true) >= 1e-8
    if not np.any(usable):
        mape = float("nan")
        excluded = int(y_true.size)
    else:
        mape = float(np.mean(np.abs(error[usable] / y_true[usable])) * 100.0)
        excluded = int((~usable).sum())
    return {
        "mae": mae,
        "mape_percent": mape,
        "rmse": rmse,
        "mse": mse,
        "n_samples": int(y_true.size),
        "mape_excluded": excluded,
        "soh_scale": "fraction_of_nominal_capacity",
        "mape_unit": "percentage",
        "aggregation": "sample_weighted_mean_over_evaluated_cycles",
    }
