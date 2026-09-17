# Historical MIT reproduction

Prepared 17 September 2026. This is an archival implementation reconstruction, kept separate from the current-code baseline. The historical full experiment is stored in `results/reproduction/MIT/historical_run_01/`. The original current-code `run_01/` is retained without changes.

## Historical source and evidence

The executable source snapshot is commit **`d72414ac8799747db784203b4c1826ca118dc23a`**, dated 23 April 2024. Seven files were extracted directly as Git blob bytes into [`historical_reproduction/source/`](../historical_reproduction/source/):

- `main_MIT.py`
- `Model/Model.py` and `Model/__init__.py`
- `dataloader/dataloader.py` and `dataloader/__init__.py`
- `utils/util.py` and `utils/__init__.py`

No snapshot source file was edited. SHA-256 hashes are recorded in [`source_manifest.json`](../historical_reproduction/source_manifest.json) and checked by the runner. The current repository files are not replaced, reverted, or monkey-patched.

Why this snapshot was selected:

1. It contains the original MIT defaults alpha=1 and beta=50, agreeing with all ten archived experiment logs and Supplementary Note 3.
2. Its physics loss is the original label-only expression. The archived epoch-level physics losses stay at 0.043094; calculating this expression on the unchanged MIT training labels gives 0.043093645347739164 as the average of batch sums.
3. The archive's predictions reproduce the published supplementary MIT results under the repository's historical analysis convention. The investigation and Git chronology are in [LOSS_WEIGHT_DISCREPANCY.md](LOSS_WEIGHT_DISCREPANCY.md).
4. This snapshot predates the July and October 2024 changes to the physics expression.

The logs describe training in July 2023, while the available public source was uploaded in April 2024. They do not record the original executable's commit. This is therefore the best-supported reconstruction from released source and numerical evidence, not proof of an identical July 2023 executable or bitwise-identical historical environment.

## Deliberately preserved label-only gradient defect

The unedited historical model computes:

```python
loss1 = 0.5*MSE(u1, y1) + 0.5*MSE(u2, y2)
loss2 = 0.5*MSE(f1, 0) + 0.5*MSE(f2, 0)
loss3 = ReLU(y2 - y1).sum()
loss = loss1 + 1*loss2 + 50*loss3
```

`y1` and `y2` are measured labels. They do not depend on the model weights, so **loss3 contributes no gradient to either network**. The effective weight updates come from data and PDE losses. The historical physics term adds approximately 2.154682 to the epoch-average total loss, regardless of how well the network fits the data. A total loss near 2.15 is thus expected and does not represent a large SOH prediction error.

This defect was preserved intentionally to reconstruct the archived numerical implementation. It must not be interpreted as successful enforcement of physical monotonicity. The paper's written prediction-based penalty, the July 2024 correction, and the current regeneration-aware penalty are different protocols and were not substituted here.

## Exact configuration

| Item | Historical run setting |
| --- | --- |
| Full experiments | Exactly one |
| Maximum epochs | 200 |
| Early-stop argument | 20, from archived logs |
| Stop condition | Original counter `>20`: 21 consecutive epochs without strict validation improvement |
| Batch size | 512 |
| Alpha / beta | 1 / 50 |
| Optimizers | Original separate Adam optimizers, original defaults |
| Solution learning rate | Initial 0.002; 30 warm-up schedule entries to 0.01, then original cosine schedule toward 0.0002 |
| Dynamics learning rate | Fixed 0.001 |
| Scheduler timing | Once after each epoch, unchanged |
| Normalization | Per-battery min–max, unchanged |
| Label | Capacity / 1.1 Ah, unchanged |
| Input features / architecture | Original 17 inputs and original networks |
| Split | Existing MIT battery-ID rule, unsorted file order; pair-level 80/20 split with random_state=420 |
| Samples | 54,053 training pairs; 13,514 validation pairs; 14,170 test pairs |
| Training RNG | No global seed, matching the archived source; initial RNG states saved |
| Environment | Existing Python 3.12.14 / PyTorch 2.14.0+cpu compatibility environment |

The April 2024 script defaulted to early_stop=10, whereas every archived MIT log records 20. The new runner explicitly supplies 20 to reconstruct the archived experiment configuration. Its output location and single-experiment scope are also explicit overrides. It does not call the historical script's inactive `__main__` guard or its ten-experiment `main()`.

## Isolation, instrumentation and checks

[`historical_reproduction/run_mit.py`](../historical_reproduction/run_mit.py) puts the historical source first on Python's import path and asserts the actual origin of model, loader, main-script and utility imports. It calls the snapshot's inherited `PINN.Train` without reimplementing its training loop. The runner reuses only the read-only file-list manifest helper from the existing smoke-test script; model and dataloader resolution remain historical.

A recording subclass observes the original epoch/validation/test return values. It adds no training forwards, backpropagation steps, batch shuffles or validation passes. It preserves the original test-at-validation-improvement timing. Loss history uses the historical equal-weight average over batch losses; validation history is the original full-loader SOH MSE.

At each validation improvement, an additional `best_checkpoint.pth` is immediately serialized. The original deferred `model.pth` is retained separately. The immediate checkpoint addresses the known shallow-state saving problem without changing optimization or selection. A final reload and test check verifies correspondence with the saved predictions. Standard ground-truth-denominator MAPE is reported separately from the historical reversed-argument MAPE.

The runner checks:

- Snapshot hashes against the extracted Git blobs and actual import paths.
- Expected dataset sizes, finite inputs, and exact hashes of all train/validation/test tensors against current-code Run 01.
- Successful optimizer updates and finite training/validation/test values.
- The label-only term does not require gradients, and its epoch average matches the fixed analytical value.
- Best-checkpoint reload reproduces saved predictions.
- SHA-256 preservation of all 26 protected current-source and current-run files, recorded before reconstruction.

The new plotting process uses a writable `MPLCONFIGDIR` inside its own output directory. This avoids the earlier external Matplotlib cache permission issue; it changes no numerical training behavior. The known pandas dtype-assignment FutureWarning remains visible and requires no source edit with the pinned environment.

## Historical smoke test

A separate three-epoch smoke test completed before the full run, with the same snapshot, data and loss weights. Outputs are in `results/smoke_tests/historical_mit_01/`; it is not counted among the full historical experiments. It verified actual parameter changes, constant physics loss, finite values, validation/test execution and exact checkpoint reload.

Preliminary smoke metrics over 14,170 test samples: MAE 0.0240802672, MAPE 2.549237%, RMSE 0.0274436363, MSE 0.0007531532. Total runtime was 13.32 seconds. These measure a three-epoch execution test, not reproduction accuracy.

## Full historical run and reference comparison

Completed successfully after **100 epochs**. The best validation checkpoint was from **epoch 79**, with validation MSE **0.0000673280956**. The original counter stopped training after 21 subsequent non-improving epochs. The training loop took **240.55 seconds**; total runner time, including loading, checkpoint verification and plotting, was **246.65 seconds (4 min 7 sec)**.

At epoch 85, training suffered a sharp but finite loss spike. Data/PDE losses subsequently decreased, but validation did not recover its previous best before early stopping. The selected checkpoint predates the spike. The separate post-training reload produced **zero difference** from the saved best-epoch predictions. This run used no test-based selection or hyperparameter changes.

### Standard metrics: same calculation for this run and archive

Metrics are sample-weighted over 14,170 test samples, with SOH as a fraction and MAPE using the ground-truth denominator. No targets are zero. Archived statistics come from the original ten stored prediction sets, not from the current-code baseline.

| Metric | historical_run_01 | Archived 10-run mean | Archived minimum–maximum | Relative error increase over mean |
| --- | ---: | ---: | ---: | ---: |
| MAE | **0.0070494441** | 0.0060177859 | 0.0057499283–0.0065251999 | 17.14% |
| MAPE (%) | **0.74300128** | 0.63574280 | 0.60766102–0.68923198 | 16.87% |
| RMSE | **0.0093178301** | 0.0082761733 | 0.0078803733–0.0090744452 | 12.59% |
| MSE | **0.0000868220** | 0.0000686079 | 0.0000621003–0.0000823456 | 26.55% |

MAE is approximately 0.705 SOH percentage points, and RMSE is approximately 0.932 SOH percentage points. The original logger's sample-weighted reversed-argument MAPE is also retained: 0.00744792493 as a fraction, or 0.744792493%.

### Historical analysis convention

For transparency, the comparison script also applies the old analysis convention to both this run and all ten archives: equal weighting across inferred battery segments, its original boundary omissions, and prediction-denominator MAPE. Under that convention:

| Metric | historical_run_01 | Archived mean |
| --- | ---: | ---: |
| MAE | 0.0073810264 | 0.0063167480 |
| MAPE (%) | 0.78071232 | 0.66732147 |
| RMSE | 0.0085437342 | 0.0075101520 |
| MSE | 0.0000927367 | 0.0000729874 |

These values are not interchangeable with the standard sample-weighted metrics. The historical MSE is an additionally calculated diagnostic under the same segmentation; the original analysis exports emphasize MAE/MAPE/RMSE.

The read-only [`compare_results.py`](../historical_reproduction/compare_results.py) confirms that all ten archived numeric argument sets match this run, all archived ordered train/test file lists match, and all archived test target arrays are bitwise identical to this run's targets. Its findings, per-run archive metrics and dispersion are saved in `archive_comparison.json`.

## Saved artifacts

All full-run outputs are in [`historical_run_01/`](../results/reproduction/MIT/historical_run_01/):

- `run_configuration.json`: historical commit, imported source paths, full settings, runtime versions, RNG policy, split lists, input tensor hashes and constant-loss expectation.
- `loss_history.csv` and `loss_history.png`: all 100 epochs, data/PDE/physics/total training losses and validation MSE, learning rates and epoch durations.
- `best_checkpoint.pth`: actual epoch-79 snapshot. Use this for the best model.
- `model.pth`: original deferred checkpoint artifact, retained for traceability; do not confuse it with the best snapshot.
- `true_label.npy`, `pred_label.npy`, `predicted_vs_true_SOH.png`: best-validation epoch test results and plot.
- `metrics.json`: metrics, best/stopping epochs, runtime, optimizer-update check and exact checkpoint-reload verification.
- `test_history.json`, `logging.txt`, `console.log`: original validation-improvement test evaluations and logs.
- `rng_initial.pth`, `dataset_sha256.json`, `requirements-reproduction.txt`, `runner_source.py`, `source_diff.patch`, `source_status.txt`: reproducibility records. The RNG file is a locally generated trusted Python-object serialization.
- `preservation_check.json`: all 26 protected current-source/Run 01 files unchanged.
- `archive_comparison.json`: archive comparisons and audit checks.

Both plots were opened and visually inspected. The loss plot displays the fixed label-only component and the epoch-85 spike. Predictions broadly track SOH, but deviations and outliers remain. No missing artifact or numerical failure was found.

## Should the remaining nine historical runs start?

**Not yet on the basis of numerical agreement.** This is a valid and carefully isolated historical-implementation run, but all four standard test errors lie above the observed archived ten-run range. MAPE is about 16.9% higher and RMSE about 12.6% higher than the archive mean. It would be premature to call the historical accuracy reproduced or to assume the remaining nine runs will resolve the gap.

One stochastic run is not enough to establish a distributional mismatch. The result does not prove a reconstruction error: initialization is unseeded, the archive lacks RNG states and the original executable, and the modern dependency stack differs substantially. The finite optimization spike is observed evidence of instability, but its cause has not been isolated. Early stopping can also occur in the archive (for example one archived run stopped at epoch 86), so stopping early is not itself invalid.

Recommended next step: audit the optimization spike and modern-versus-historical numerical environment before launching the remaining batch. If a controlled diagnostic training comparison is later authorized, predefine its protocol and keep all results; do not select seeds or tune on the held-out test errors to force agreement. Keep the completed historical run and current-code Run 01 intact. No additional full experiments were started.
