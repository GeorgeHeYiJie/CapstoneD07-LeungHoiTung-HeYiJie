"""Convert MIT and HUST cycle tables into Wu et al. (2023) 2D matrices.

One battery becomes one ``.npz`` file. The original CSVs are not modified.

Example, from the repository root:

    python scripts/convert_wu_matrix.py --dataset both --k 5 --output-dir results/wu_matrix
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wu_matrix.datasets import FEATURE_COLUMNS, list_batteries, read_battery_csv
from wu_matrix.transform import series_to_matrices


def convert_dataset(dataset: str, data_root: str, output_dir: str, k: int) -> list[dict]:
    destination = os.path.join(output_dir, dataset)
    os.makedirs(destination, exist_ok=True)
    manifest = []
    for record in list_batteries(dataset, data_root):
        features, capacity = read_battery_csv(record["path"])
        converted = series_to_matrices(features, k=k)
        out_path = os.path.join(destination, record["battery_id"] + ".npz")
        if os.path.exists(out_path):
            raise FileExistsError(out_path)
        payload = {
            "capacity": capacity,
            "periods": converted["periods"],
            "frequencies": converted["frequencies"],
            "amplitudes": converted["amplitudes"],
            "n_cycles": np.array(converted["n_cycles"]),
            "feature_names": np.array(FEATURE_COLUMNS),
        }
        for index, matrix in enumerate(converted["matrices"]):
            payload[f"matrix_{index}"] = matrix
        np.savez(out_path, **payload)
        manifest.append(
            {
                "dataset": dataset,
                "battery_id": record["battery_id"],
                "condition": record["condition"],
                "source_csv": record["path"],
                "output_npz": out_path,
                "n_cycles": converted["n_cycles"],
                "n_features": converted["n_features"],
                "periods": converted["periods"].tolist(),
                "frequencies": converted["frequencies"].tolist(),
                "matrix_shapes": [list(matrix.shape) for matrix in converted["matrices"]],
                "layout": "[n_periods, period, features]",
            }
        )
        print(
            f"{dataset} {record['battery_id']}: T={converted['n_cycles']} "
            f"periods={converted['periods'].tolist()}",
            flush=True,
        )
    manifest_path = os.path.join(destination, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "k": k,
                "feature_columns": FEATURE_COLUMNS,
                "label": "capacity, saved beside the matrices and not used as an FFT channel",
                "layout": "matrix_i has shape [n_periods, period, features]; rows are successive periods, columns are the phase inside one period",
                "padding": "zeros appended on the cycle axis so the length is divisible by the period",
                "batteries": manifest,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Wu et al. 2023 multi-period matrix conversion for MIT and HUST")
    parser.add_argument("--dataset", choices=["MIT", "HUST", "both"], default="both")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/wu_matrix")
    parser.add_argument("--k", type=int, default=5, help="number of FFT periods")
    args = parser.parse_args()
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    datasets = ["MIT", "HUST"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        convert_dataset(dataset, args.data_root, args.output_dir, args.k)


if __name__ == "__main__":
    main()
