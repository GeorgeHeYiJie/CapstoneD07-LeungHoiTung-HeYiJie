"""MIT and HUST loading for cross-condition SOH.

Cleaning and the nominal-capacity divisor follow Wang et al. (2024):
drop non-finite rows, drop 3-sigma outliers on every column, then divide
capacity by 1.1 Ah. Cycle index is the row number before that deletion.

Two choices are deliberately different from the PINN loader, because Zhang's
inter-cell subtraction needs a shared scale and an unseen condition:

* Min-max to [-1, 1] is fit on the training batteries only, then applied to
  validation and test. It is not fit separately on each battery.
* The test set is an entire MIT date-batch or an entire HUST filename group.
  Validation is a disjoint 20% of the remaining batteries (``random_state=420``),
  not a random split of adjacent pairs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
NOMINAL_CAPACITY_AH = {"MIT": 1.1, "HUST": 1.1}
MIT_BATCHES = ("2017-05-12", "2017-06-30", "2018-04-12")


@dataclass
class BatteryRecord:
    battery_id: str
    condition: str
    path: str
    cycle_index: np.ndarray
    features: np.ndarray
    soh: np.ndarray
    features_norm: np.ndarray | None = None


@dataclass
class FeatureScaler:
    minimum: np.ndarray
    maximum: np.ndarray
    zero_range_columns: list[int] = field(default_factory=list)

    def transform(self, features_17: np.ndarray) -> np.ndarray:
        span = self.maximum - self.minimum
        safe = span.copy()
        safe[safe == 0] = 1.0
        scaled = 2.0 * (features_17 - self.minimum) / safe - 1.0
        if self.zero_range_columns:
            scaled[..., self.zero_range_columns] = 0.0
        return scaled.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "zero_range_columns": self.zero_range_columns,
            "method": "min-max to [-1, 1], fit on training batteries only",
        }


@dataclass
class SampleIndex:
    battery_pos: int
    cycle_pos: int
    next_cycle_pos: int


def delete_3_sigma(frame: pd.DataFrame) -> pd.DataFrame:
    """Same outlier rule as ``dataloader.dataloader.DF.delete_3_sigma``."""
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    outlier_rows: list[int] = []
    for column in frame.columns:
        series = frame[column]
        std = series.std()
        if not np.isfinite(std):
            continue
        mean = series.mean()
        rule = (mean - 3 * std > series) | (mean + 3 * std < series)
        outlier_rows.extend(np.flatnonzero(rule.to_numpy()))
    if outlier_rows:
        frame = frame.drop(index=sorted(set(outlier_rows))).reset_index(drop=True)
    return frame


def read_battery_table(path: str, nominal_capacity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cycle index, 16 raw features and SOH for one CSV."""
    frame = pd.read_csv(path)
    missing = [name for name in FEATURE_COLUMNS + ["capacity"] if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    frame = frame[FEATURE_COLUMNS + ["capacity"]].copy()
    frame.insert(frame.shape[1] - 1, "cycle index", np.arange(frame.shape[0]))
    frame = delete_3_sigma(frame)
    if len(frame) < 2:
        raise ValueError(f"{path} has fewer than 2 rows after cleaning")
    cycle_index = frame["cycle index"].to_numpy(dtype=np.float64)
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    soh = frame["capacity"].to_numpy(dtype=np.float64) / float(nominal_capacity)
    return cycle_index, features, soh


def list_mit_files(root: str = "data/MIT data") -> list[tuple[str, str, str]]:
    """Return ``(battery_id, batch, path)`` sorted by battery id."""
    rows = []
    for batch in MIT_BATCHES:
        batch_root = os.path.join(root, batch)
        if not os.path.isdir(batch_root):
            raise FileNotFoundError(batch_root)
        for name in os.listdir(batch_root):
            if not name.endswith(".csv"):
                continue
            battery_id = name[:-4]
            rows.append((battery_id, batch, os.path.join(batch_root, name)))
    return sorted(rows, key=lambda item: item[0])


def list_hust_files(root: str = "data/HUST data") -> list[tuple[str, str, str]]:
    """Return ``(battery_id, group, path)``.

    The group is the filename prefix before the hyphen. It is the only
    condition label stored with these preprocessed tables.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    rows = []
    for name in os.listdir(root):
        if not name.endswith(".csv"):
            continue
        battery_id = name[:-4]
        group = battery_id.split("-")[0]
        rows.append((battery_id, group, os.path.join(root, name)))
    return sorted(rows, key=lambda item: (int(item[1]), int(item[0].split("-")[1]), item[0]))


def list_dataset(dataset: str, root: str | None = None) -> list[tuple[str, str, str]]:
    if dataset == "MIT":
        return list_mit_files(root or "data/MIT data")
    if dataset == "HUST":
        return list_hust_files(root or "data/HUST data")
    raise ValueError("dataset must be MIT or HUST")


def condition_names(dataset: str) -> list[str]:
    files = list_dataset(dataset)
    if dataset == "MIT":
        return list(MIT_BATCHES)
    return sorted({condition for _, condition, _ in files}, key=lambda value: int(value))


def split_batteries(
    files: list[tuple[str, str, str]],
    held_out_condition: str,
    val_ratio: float = 0.2,
    seed: int = 420,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> dict:
    """Hold out one condition. Split the other batteries into train and val.

    Caps, when set, keep the alphabetically first ids of that split. They are
    a smoke-test bound. The uncapped membership is retained in the manifest.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    known = sorted({condition for _, condition, _ in files})
    if held_out_condition not in known:
        raise ValueError(f"held-out condition {held_out_condition!r} is not one of {known}")
    test = [item for item in files if item[1] == held_out_condition]
    rest = [item for item in files if item[1] != held_out_condition]
    if len(rest) < 2:
        raise ValueError("need at least two batteries outside the held-out condition")
    train, val = train_test_split(rest, test_size=val_ratio, random_state=seed)
    train = sorted(train, key=lambda item: item[0])
    val = sorted(val, key=lambda item: item[0])
    test = sorted(test, key=lambda item: item[0])

    def cap(rows, limit):
        if limit is None:
            return rows
        if limit < 1:
            raise ValueError("battery caps must be positive")
        return rows[:limit]

    used_train = cap(train, max_train)
    used_val = cap(val, max_val)
    used_test = cap(test, max_test)
    if len(used_train) < 2:
        raise ValueError("training needs at least 2 batteries so a reference cell exists")
    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "held_out_condition": held_out_condition,
        "full_train": train,
        "full_val": val,
        "full_test": test,
        "train": used_train,
        "val": used_val,
        "test": used_test,
    }


def _load_rows(rows: list[tuple[str, str, str]], nominal_capacity: float) -> list[BatteryRecord]:
    records = []
    for battery_id, condition, path in rows:
        cycle_index, features, soh = read_battery_table(path, nominal_capacity)
        records.append(
            BatteryRecord(
                battery_id=battery_id,
                condition=condition,
                path=path,
                cycle_index=cycle_index,
                features=features,
                soh=soh.astype(np.float64),
            )
        )
    return records


def fit_scaler(records: list[BatteryRecord]) -> FeatureScaler:
    columns = []
    for record in records:
        cycle = record.cycle_index.reshape(-1, 1)
        columns.append(np.concatenate([record.features, cycle], axis=1))
    stacked = np.concatenate(columns, axis=0)
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    zero_range = np.flatnonzero(np.isclose(maximum, minimum)).tolist()
    return FeatureScaler(minimum=minimum, maximum=maximum, zero_range_columns=zero_range)


def apply_scaler(records: list[BatteryRecord], scaler: FeatureScaler) -> None:
    for record in records:
        raw = np.concatenate([record.features, record.cycle_index.reshape(-1, 1)], axis=1)
        record.features_norm = scaler.transform(raw)


def build_windows_index(records: list[BatteryRecord], stride: int, require_next: bool) -> list[SampleIndex]:
    if stride < 1:
        raise ValueError("stride must be positive")
    samples = []
    for pos, record in enumerate(records):
        length = len(record.soh)
        last = length - 2 if require_next else length - 1
        for cycle_pos in range(0, last + 1, stride):
            next_pos = cycle_pos + 1 if require_next else -1
            samples.append(SampleIndex(pos, cycle_pos, next_pos))
    if not samples:
        raise ValueError("no evaluation cycles were produced")
    return samples


def make_window(features_norm: np.ndarray, cycle_pos: int, window: int) -> np.ndarray:
    """Causal window ending at ``cycle_pos``.

    Cycles before the window starts are filled by repeating the first retained
    cycle. Wu's own zero padding is applied later, inside the period reshape.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    length = features_norm.shape[0]
    if cycle_pos < 0 or cycle_pos >= length:
        raise IndexError(f"cycle {cycle_pos} outside 0..{length - 1}")
    start = cycle_pos - window + 1
    if start >= 0:
        return features_norm[start : cycle_pos + 1]
    pad = np.repeat(features_norm[:1], -start, axis=0)
    return np.concatenate([pad, features_norm[: cycle_pos + 1]], axis=0)


def stack_batch(
    target_records: list[BatteryRecord],
    samples: list[SampleIndex],
    reference_positions: list[int],
    reference_records: list[BatteryRecord],
    window: int,
) -> dict[str, np.ndarray]:
    """Build target and reference windows for a list of samples.

    The reference sequence is aligned by retained-row index. If the reference
    cell is shorter, its last retained cycle is reused.
    """
    windows = []
    next_windows = []
    refs = []
    next_refs = []
    y = []
    y_next = []
    y_ref = []
    y_ref_next = []
    x0 = []
    battery_ids = []
    conditions = []
    cycle_indexes = []
    has_next = samples[0].next_cycle_pos >= 0
    for sample, ref_pos in zip(samples, reference_positions):
        record = target_records[sample.battery_pos]
        reference = reference_records[ref_pos]
        if record.features_norm is None or reference.features_norm is None:
            raise RuntimeError("normalize features before building windows")
        windows.append(make_window(record.features_norm, sample.cycle_pos, window))
        x0.append(record.features_norm[0])
        y.append(record.soh[sample.cycle_pos])
        ref_cycle = min(sample.cycle_pos, len(reference.soh) - 1)
        refs.append(make_window(reference.features_norm, ref_cycle, window))
        y_ref.append(reference.soh[ref_cycle])
        battery_ids.append(record.battery_id)
        conditions.append(record.condition)
        cycle_indexes.append(record.cycle_index[sample.cycle_pos])
        if has_next:
            next_pos = sample.next_cycle_pos
            next_windows.append(make_window(record.features_norm, next_pos, window))
            y_next.append(record.soh[next_pos])
            ref_next = min(next_pos, len(reference.soh) - 1)
            next_refs.append(make_window(reference.features_norm, ref_next, window))
            y_ref_next.append(reference.soh[ref_next])
    batch = {
        "window": np.stack(windows).astype(np.float32),
        "ref_window": np.stack(refs).astype(np.float32),
        "y": np.asarray(y, dtype=np.float32).reshape(-1, 1),
        "y_ref": np.asarray(y_ref, dtype=np.float32).reshape(-1, 1),
        "x0": np.stack(x0).astype(np.float32),
        "battery_id": np.asarray(battery_ids),
        "condition": np.asarray(conditions),
        "cycle_index": np.asarray(cycle_indexes, dtype=np.float64),
    }
    if has_next:
        batch["window_next"] = np.stack(next_windows).astype(np.float32)
        batch["ref_window_next"] = np.stack(next_refs).astype(np.float32)
        batch["y_next"] = np.asarray(y_next, dtype=np.float32).reshape(-1, 1)
        batch["y_ref_next"] = np.asarray(y_ref_next, dtype=np.float32).reshape(-1, 1)
    return batch


def load_split_records(dataset: str, split: dict) -> dict[str, list[BatteryRecord]]:
    nominal = NOMINAL_CAPACITY_AH[dataset]
    return {
        "train": _load_rows(split["train"], nominal),
        "val": _load_rows(split["val"], nominal),
        "test": _load_rows(split["test"], nominal),
    }


def manifest_ids(rows: list[tuple[str, str, str]]) -> list[dict]:
    return [{"battery_id": battery_id, "condition": condition, "path": path} for battery_id, condition, path in rows]
