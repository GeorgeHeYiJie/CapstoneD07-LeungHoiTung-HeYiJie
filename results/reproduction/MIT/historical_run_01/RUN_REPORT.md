# Historical MIT run 01

Completed 17 September 2026. One full historical experiment, preceded by a separate three-epoch smoke test. No runs 02–10 started.

Source snapshot: `d72414ac8799747db784203b4c1826ca118dc23a`, byte-for-byte isolated under `historical_reproduction/source/`. Settings: alpha=1, beta=50, 200-epoch maximum, batch size 512 and archived early_stop=20. Original architecture, split, features, labels, optimizers, scheduler timing and training loop retained.

**Intentional historical defect:** physics loss is `ReLU(y2-y1).sum()`, involving measured labels only. It has no model-parameter gradient. Epoch-average physics loss remains approximately 0.04309364535, consistent with the archived logs.

| Outcome | Value |
| --- | ---: |
| Best epoch | 79 |
| Stopping epoch | 100 |
| Best validation MSE | 0.0000673280956 |
| Test MAE | 0.0070494441 |
| Test MAPE | 0.74300128% |
| Test RMSE | 0.0093178301 |
| Test MSE | 0.0000868220 |
| Test samples | 14,170 |
| Training runtime | 240.55 seconds |
| Total runner runtime | 246.65 seconds |
| Reloaded best-checkpoint prediction difference | 0.0 |

Metrics are sample-weighted with fractional SOH and ground-truth-denominator MAPE. Best checkpoint is `best_checkpoint.pth`; `model.pth` preserves the author's deferred-saving behavior and is not the selected best snapshot.

Archive means using the same metric calculation: MAE 0.0060177859, MAPE 0.63574280%, RMSE 0.0082761733, MSE 0.0000686079. All four current errors exceed the observed archived range. A finite loss spike occurred at epoch 85; validation did not recover its epoch-79 minimum before original early stopping at 100.

Technical checks passed: source provenance, archived arguments and ordered file lists, dataset tensors, finite losses/predictions, optimizer update, label-only physics term, correct early stopping and exact best-checkpoint reload. All protected current model/current-run files remain unchanged.

**Recommendation:** do not launch the remaining nine yet as a validated numerical reproduction. First investigate the optimization instability and numerical environment differences. This one stochastic result does not by itself prove a code error or a distributional mismatch. Preserve it without rerunning or selecting favorable seeds.

See `archive_comparison.json` for full reference statistics and `docs/HISTORICAL_REPRODUCTION.md` in the repository for reconstruction details and limitations. Loss histories, plots, predictions, configuration, runtime, RNG state, hashes and logs are stored alongside this report.
