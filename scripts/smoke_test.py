"""Run one bounded, three-epoch smoke test of the authors' original PINN."""

from argparse import Namespace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from main_MIT import load_MIT_data  # noqa: E402
from Model.Model import PINN  # noqa: E402


SEED = 420
FEATURE_NAMES = [
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
]


def make_args(output_directory: Path) -> Namespace:
    """Match main_MIT.py defaults except for the requested three epochs."""
    return Namespace(
        data="MIT",
        batch_size=512,
        normalization_method="min-max",
        epochs=3,
        early_stop=20,
        warmup_epochs=30,
        warmup_lr=2e-3,
        lr=1e-2,
        final_lr=2e-4,
        lr_F=1e-3,
        u_layers_num=3,
        u_hidden_dim=60,
        F_layers_num=3,
        F_hidden_dim=60,
        alpha=0.5,
        beta=0.01,
        log_dir="logging.txt",
        save_folder=str(output_directory),
    )


def mit_split_manifest() -> dict:
    """Record the file split selected by the unchanged main_MIT loader logic."""
    training_files = []
    test_files = []
    data_root = REPOSITORY_ROOT / "data" / "MIT data"
    for batch in ["2017-05-12", "2017-06-30", "2018-04-12"]:
        batch_root = data_root / batch
        for filename in os.listdir(batch_root):
            battery_id = int(filename.split("-")[-1].split(".")[0])
            relative_path = (Path("data") / "MIT data" / batch / filename).as_posix()
            if battery_id % 5 == 0:
                test_files.append(relative_path)
            else:
                training_files.append(relative_path)
    return {
        "rule": "Test battery ID is divisible by 5; all other batteries feed the existing pair-level 80/20 train/validation split with random_state=420.",
        "training_and_validation_battery_files": training_files,
        "test_battery_files": test_files,
    }


def main() -> int:
    os.chdir(REPOSITORY_ROOT)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_id = datetime.now(timezone.utc).strftime("mit_%Y%m%dT%H%M%SZ")
    output_directory = REPOSITORY_ROOT / "results" / "smoke_tests" / run_id
    output_directory.mkdir(parents=True, exist_ok=False)
    args = make_args(output_directory)

    print("Dataset: MIT (processed feature CSVs included in the repository)")
    print("Dataloader: main_MIT.load_MIT_data -> dataloader.MITdata")
    print("Experiment count: 1")
    print("Epoch count: 3")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {output_directory}")

    loaders = load_MIT_data(args)
    train_x1, _, train_y1, _ = loaders["train"].dataset.tensors
    print(f"Training input shape: {tuple(train_x1.shape)}")
    print(f"Target shape: {tuple(train_y1.shape)}")
    print("First sample's 17 input values:")
    print(train_x1[0].tolist())
    print("Cycle/time input: input 17 (zero-based column 16), named 'cycle index'.")

    if train_x1.ndim != 2 or train_x1.shape[1] != 17:
        raise RuntimeError(f"Expected 17 model inputs, got shape {tuple(train_x1.shape)}")
    if train_y1.ndim != 2 or train_y1.shape[1] != 1:
        raise RuntimeError(f"Expected one target value, got shape {tuple(train_y1.shape)}")
    if not torch.isfinite(train_x1).all() or not torch.isfinite(train_y1).all():
        raise RuntimeError("Training input or target contains a non-finite value.")

    model = PINN(args)
    epoch_records = []
    for epoch in range(1, args.epochs + 1):
        data_loss, pde_loss, physics_loss = model.train_one_epoch(epoch, loaders["train"])
        learning_rate = model.scheduler.step()
        total_loss = data_loss + args.alpha * pde_loss + args.beta * physics_loss
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "data_loss": data_loss,
            "pde_loss": pde_loss,
            "physics_loss": physics_loss,
            "total_loss": total_loss,
        }
        epoch_records.append(record)
        print(f"Epoch {epoch}/3")
        print(f"  data loss:    {data_loss:.10f}")
        print(f"  PDE loss:     {pde_loss:.10f}")
        print(f"  physics loss: {physics_loss:.10f}")
        print(f"  total loss:   {total_loss:.10f}")

    true_soh, predicted_soh = model.Test(loaders["test"])
    if not np.isfinite(predicted_soh).all():
        raise RuntimeError("Test predictions contain a non-finite value.")
    mae = mean_absolute_error(true_soh, predicted_soh)
    mape_fraction = mean_absolute_percentage_error(true_soh, predicted_soh)
    mse = mean_squared_error(true_soh, predicted_soh)
    rmse = float(np.sqrt(mse))
    metrics = {
        "MAE": float(mae),
        "MAPE_fraction": float(mape_fraction),
        "MAPE_percent": float(100 * mape_fraction),
        "RMSE": rmse,
        "MSE": float(mse),
        "test_samples": int(true_soh.shape[0]),
    }
    print("Test metrics (sample-weighted; SOH is a fraction):")
    print(f"  MAE:  {metrics['MAE']:.10f}")
    print(f"  MAPE: {metrics['MAPE_percent']:.6f}%")
    print(f"  RMSE: {metrics['RMSE']:.10f}")
    print(f"  MSE:  {metrics['MSE']:.10f}")

    np.save(output_directory / "true_label.npy", true_soh)
    np.save(output_directory / "pred_label.npy", predicted_soh)
    torch.save(
        {
            "solution_u": model.solution_u.state_dict(),
            "dynamical_F": model.dynamical_F.state_dict(),
            "epoch": args.epochs,
            "checkpoint_selection": "final epoch of smoke test",
        },
        output_directory / "model_final_epoch.pth",
    )
    manifest = {
        "status": "smoke test only; not a paper reproduction result",
        "run_id": run_id,
        "seed": SEED,
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(next(model.parameters()).device),
        "feature_names_in_order": FEATURE_NAMES,
        "target": "capacity / 1.1 Ah nominal capacity (SOH fraction)",
        "arguments": vars(args),
        "split": mit_split_manifest(),
        "dataset_sizes": {name: len(loader.dataset) for name, loader in loaders.items()},
        "losses": epoch_records,
        "metrics": metrics,
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    model.clear_logger()
    print("Smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
