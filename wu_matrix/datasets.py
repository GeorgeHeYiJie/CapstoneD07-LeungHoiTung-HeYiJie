"""Read MIT and HUST cycle tables for the Wu matrix conversion.

Each CSV is already one row per cycle. Column order is the header order.
The 16 charging features are the series that gets reshaped. ``capacity``
stays beside the matrices as the cycle label and is not an FFT channel.
Rows are kept in file order. This step does not apply the PINN cleaning
or per-battery normalization.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

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
LABEL_COLUMN = "capacity"
MIT_BATCHES = ("2017-05-12", "2017-06-30", "2018-04-12")


def list_batteries(dataset: str, data_root: str = "data") -> list[dict]:
    """Return sorted records with battery_id, condition and path."""
    if dataset == "MIT":
        root = os.path.join(data_root, "MIT data")
        rows = []
        for batch in MIT_BATCHES:
            batch_root = os.path.join(root, batch)
            if not os.path.isdir(batch_root):
                raise FileNotFoundError(batch_root)
            for name in os.listdir(batch_root):
                if name.endswith(".csv"):
                    rows.append(
                        {
                            "dataset": "MIT",
                            "battery_id": name[:-4],
                            "condition": batch,
                            "path": os.path.join(batch_root, name),
                        }
                    )
        return sorted(rows, key=lambda item: item["battery_id"])
    if dataset == "HUST":
        root = os.path.join(data_root, "HUST data")
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        rows = []
        for name in os.listdir(root):
            if not name.endswith(".csv"):
                continue
            battery_id = name[:-4]
            rows.append(
                {
                    "dataset": "HUST",
                    "battery_id": battery_id,
                    "condition": battery_id.split("-")[0],
                    "path": os.path.join(root, name),
                }
            )
        return sorted(rows, key=lambda item: (int(item["condition"]), int(item["battery_id"].split("-")[1])))
    raise ValueError("dataset must be MIT or HUST")


def read_battery_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return features [T, 16] and capacity [T] from one CSV."""
    frame = pd.read_csv(path)
    missing = [name for name in FEATURE_COLUMNS + [LABEL_COLUMN] if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    capacity = frame[LABEL_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(capacity).all():
        raise ValueError(f"{path} contains NaN or infinity")
    if len(features) < 2:
        raise ValueError(f"{path} has fewer than 2 cycles")
    return features, capacity
