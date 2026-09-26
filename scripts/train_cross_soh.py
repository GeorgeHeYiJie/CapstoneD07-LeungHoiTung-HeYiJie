"""Train one cross-condition SOH fold.

The script runs a single held-out condition. It does not loop over MIT
batches, HUST groups, or repeated experiments.

Use ``--smoke`` for a bounded execution check. Use ``--full-fold`` only when
a complete training run on that one condition is intended.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_soh.data import condition_names
from cross_soh.train import TrainConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-condition SOH: Wu matrix + Zhang branches + Wang losses")
    parser.add_argument("--dataset", choices=["MIT", "HUST"], required=True)
    parser.add_argument("--held-out", required=True, help="MIT date batch or HUST filename group")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-refs", type=int, default=4)
    parser.add_argument("--mix-alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Cap batteries, width and epochs. Metrics are preliminary.")
    mode.add_argument("--full-fold", action="store_true", help="Use every battery in this one held-out fold.")
    return parser.parse_args()


def main():
    args = parse_args()
    known = condition_names(args.dataset)
    if args.held_out not in known:
        raise SystemExit(f"--held-out must be one of {known}")
    cfg = TrainConfig(
        dataset=args.dataset,
        held_out=args.held_out,
        save_dir=args.save_dir,
        data_root=args.data_root,
        epochs=args.epochs,
        early_stop=args.early_stop,
        batch_size=args.batch_size,
        window=args.window,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        seed=args.seed,
        n_refs=args.n_refs,
        mix_alpha=args.mix_alpha,
        device=args.device,
        smoke=args.smoke,
    )
    if args.smoke:
        # A smoke test checks that one fold runs. It is not a paper comparison.
        cfg.epochs = min(cfg.epochs, 2)
        cfg.max_train_batteries = 3
        cfg.max_val_batteries = 1
        cfg.max_test_batteries = 1
        cfg.window = 16
        cfg.train_stride = 10
        cfg.eval_stride = 10
        cfg.batch_size = 4
        cfg.d_model = 8
        cfg.n_layers = 1
        cfg.k_periods = 2
        cfg.n_kernels = 2
        cfg.n_refs = 2
        cfg.early_stop = 5
    run_experiment(cfg)


if __name__ == "__main__":
    main()
