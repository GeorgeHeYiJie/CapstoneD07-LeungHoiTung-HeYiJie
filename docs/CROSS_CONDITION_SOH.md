# Cross-condition SOH model

This is a new experiment. The PINN in `Model/Model.py` is unchanged and remains the reference baseline. Nothing here overwrites `data/`, `pretrained model/` or `results/reproduction/`.

The model estimates state of health (SOH) for MIT and HUST cells. After the CSV is read, each cell's cycle-feature sequence is turned into period-based 2D matrices (Wu et al., 2023). SOH is then predicted by a two-branch network (Zhang et al., 2025) trained with the PINN losses (Wang et al., 2024). The test batteries come from an ageing condition that is absent from training.

## What each paper contributes

| Step | Source | What this code does |
| --- | --- | --- |
| Read MIT / HUST tables | Wang et al., 2024, as stored in this repository | 16 charging features plus the row-based cycle index. Capacity is divided by 1.1 Ah. |
| Clean rows | Wang et al., 2024 loader | Drop non-finite rows, then drop 3-sigma outliers. Cycle index is assigned before that deletion and is not renumbered. |
| 1D series to 2D matrices | Wu et al., 2023, TimesNet | FFT, top-k periods, zero-pad, reshape. Rows are successive periods. Columns are the phase inside a period. |
| Read those matrices | Wu et al., 2023, TimesBlock | A small inception applies 2D kernels, then the branches are weighted by FFT amplitude. |
| Two predictors | Zhang et al., 2025, BatLiNet | Intra-cell branch: difference from the cell's first retained cycle. Inter-cell branch: difference from a training cell, plus that cell's SOH. Both branches share the last linear layer. |
| Losses | Wang et al., 2024 | Data MSE, dynamics residual, and the direction penalty `sum(ReLU((u2-u1)*(y1-y2)))`. |
| Unseen condition | Zhang et al., 2025 | The test set is a whole MIT date batch or a whole HUST filename group. |

Zhang's published model predicts cycle life from raw voltage-capacity curves. This repository stores Wang's per-cycle feature tables, not those curves, so the intra-cell and inter-cell differences are computed on the 17 normalized inputs. The target is SOH at the current cycle, not cycle life. The contrastive label is therefore the SOH difference at the aligned cycle.

Wu's paper writes `p = ceil(T / f)` for the period. The released TimesNet code uses `p = T // f`, pads with zeros, and puts one period along the last axis. This implementation follows that working procedure. `cross_soh/times_matrix.py` is the only place the reshape is defined. Each TimesBlock calls it again because the hidden sequence is a new series.

## Cross-condition split

This is not the PINN split, and it is not a claim that the cell lists match Zhang's MATR-1 or MATR-2 tables.

* **MIT.** The condition is the experiment-date folder: `2017-05-12`, `2017-06-30`, or `2018-04-12`. These are the three MATR batches. One folder is the test set.
* **HUST.** The condition is the number before the hyphen in the filename (`1` through `10`). That is the only condition label in these tables. One group is the test set. Zhang's HUST numbers use a BatteryML cell split that is not stored here.
* Of the batteries that are not in the held-out condition, 20% are validation and the rest are training. The split uses `train_test_split(..., test_size=0.2, random_state=420)` on batteries, not on adjacent pairs.
* Train, validation and test battery ids do not overlap. The held-out condition appears only in the test manifest.
* A smoke run keeps only 3 training, 1 validation and 1 test battery. The cap rotates through conditions in sorted order, so the three training batteries are not all taken from the condition whose ids sort first. The uncapped membership is still written to the manifest.
* Normalization to `[-1, 1]` is fit on the training batteries that this run actually uses, then applied to validation and test. SOH itself is not rescaled. A column with zero training range is set to 0.
* Reference cells used at validation and test are training batteries only. The reported prediction is the mean over the first `n_refs` training batteries in sorted order. Zhang's methods text averages reference predictions; one of their figures uses the median. This code uses the mean.

The PINN loader instead min-max scales each battery with its own full trajectory, including test cells, and its MIT test set is every battery id divisible by 5 from all three batches. Those two protocols answer different questions. Do not put the numbers in one table as a controlled comparison.

## Prediction rule

For a window ending at the cycle being scored:

```text
y_intra = w · h_theta(window - first_cycle)
y_inter = w · h_phi(window - reference_window) + SOH_reference
SOH_hat = mix_alpha · y_intra + (1 - mix_alpha) · y_inter
```

`mix_alpha` defaults to 0.5. `w` is one shared linear layer. The two encoders do not share weights.

Adjacent retained rows of the same cell form the training pair. A stride greater than 1 keeps every stride-th row, and the physics term still compares that row with the next retained row. The window is causal: missing early cycles are filled by repeating the first retained cycle. Zero padding is only the period-reshape padding inside the TimesBlock.

Wang's residual uses the partial derivatives of `SOH_hat` with respect to the 17 inputs of the current cycle. The dynamics network is the same width as the PINN default, `35 -> 60 -> 60 -> 1`. Period selection is discrete, so the derivative does not flow through which period was chosen.

Default loss weights are Wang's dataset values: MIT `alpha=0.5`, `beta=0.01`; HUST `alpha=0.6`, `beta=0.1`. They have not been retuned for this network. `beta` multiplies a sum, so it still grows with the batch. `lambda_delta=1` weights the inter-cell SOH-difference loss.

Checkpoints are chosen by validation MSE. Test metrics are computed only after that checkpoint is loaded. `best_model.pt` is that checkpoint. `last_model.pt` is the final epoch and is not the reported model.

## How to run

From the repository root. One command is one held-out condition. It does not repeat the run ten times.

Smoke test (a few batteries, 2 epochs at most). The metrics are preliminary:

```bash
python scripts/train_cross_soh.py --dataset MIT --held-out 2018-04-12 \
  --smoke --save-dir results/smoke_tests/cross_soh/mit_2018-04-12
```

```bash
python scripts/train_cross_soh.py --dataset HUST --held-out 10 \
  --smoke --save-dir results/smoke_tests/cross_soh/hust_group10
```

A full fold uses every battery in that split. It is a separate decision from the smoke test:

```bash
python scripts/train_cross_soh.py --dataset MIT --held-out 2018-04-12 \
  --full-fold --epochs 30 --save-dir results/cross_soh/MIT/2018-04-12/run_01
```

The save directory must be new. The run writes `config.json`, `split_manifest.json`, `metrics.json`, `history.json`, `example_period_matrices.npz`, `test_predictions.npz`, and the two checkpoints.

## Metrics

SOH is a fraction of 1.1 Ah. MAE and RMSE use that fraction. MSE is its square. MAPE is `100 * mean(|y_true - y_pred| / |y_true|)`, with `y_true` in the denominator. Samples with `|y_true| < 1e-8` are omitted from MAPE only and counted in `mape_excluded`.

`metrics.json` reports the sample-weighted test scores and the unweighted mean of the per-battery scores. Smoke runs set `"preliminary": true`.

## Preliminary smoke runs

These two runs only check that one fold executes. Each used 3 training batteries, 1 validation battery, 1 test battery, a window of 16 cycles, stride 10, and 2 epochs on CPU. The checkpoint is the best validation epoch. Validation MSE improved on epoch 2, so that epoch is also the last epoch. SOH is a fraction of 1.1 Ah. MAPE is a percentage of the ground truth. No target was near zero. The test battery is one cell, so the per-battery mean equals the sample-weighted score.

The machine was Linux, Python 3.12.3, PyTorch 2.14.0+cu130 with CUDA available but `device=cpu`, NumPy 2.4.4, pandas 3.0.6, scikit-learn 1.9.1. That is not the Windows reproduction environment in `ENVIRONMENT.md`.

| Run | Held-out test cell | Samples | MAE | MAPE (%) | RMSE | MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MIT | `2018-04-12_battery-1` | 91 | 0.123285 | 13.0487 | 0.156543 | 0.024506 |
| HUST | `10-1` | 168 | 0.164755 | 16.4404 | 0.200474 | 0.040190 |

MIT training cells were `2017-05-12_battery-10`, `2017-06-30_battery-1` and `2017-05-12_battery-11`. HUST training cells were `1-1`, `2-2` and `3-1`. The first saved period matrices had shapes `[4, 5, 17]` and `[1, 16, 17]` for MIT, and `[2, 8, 17]` and `[1, 16, 17]` for HUST: rows, period length, 17 inputs. Predictions were finite. These errors are not a comparison with the PINN baseline or with the papers.

## Limits

* A smoke test shows that the fold runs. It does not reproduce Wang's, Wu's or Zhang's published errors.
* There is no raw voltage-capacity curve in this repository, so this is not a BatLiNet reproduction.
* Cycle index is still an input, because removing it would change Wang's feature set. It can tell the model how far the recorded trajectory has progressed, which may not transfer to a new condition.
* Validation batteries share the training conditions. Only the test condition is unseen.
* The dynamics residual is a learned penalty on this network's derivative. It is not an electrochemical proof.
* Reference alignment uses the retained-row index. A shorter reference cell repeats its last cycle.
