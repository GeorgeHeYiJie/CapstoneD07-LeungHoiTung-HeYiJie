# Wang MIT comparison: three seeds

Completed seeds 42, 7 and 123 for W-O and W-H. No Zhang experiments and no FFT/no-FFT ablation were run. Existing seed-42 outputs and the detailed single-seed report remain unchanged.

## Protocol and verification

HD-SOH-v1: identical 71/18/36 train/validation/test batteries; 42,784/13,585/29,083 pairs; split seed 420; SOH=capacity/1.1; train-only min-max; batch 512; 200 epochs without early stopping; CPU with 8 threads. Each run uses its own lowest validation-MSE checkpoint. All six test key/label sequences match exactly, and saved metrics were independently recomputed from predictions.

Only training seeds changed. Python, NumPy, torch initialization/dropout and batch-order randomness follow each seed; the split is unchanged. The original helper and model code were not modified. Wang losses and both Adam optimizers/scheduler remain unchanged. The user-selected seeds 7 and 123 supersede the tentative 43/44 listed in the initial plan.

MAE/RMSE are SOH fractions; MSE is squared SOH fraction; MAPE is a percentage with true SOH as denominator. Reported SD is sample SD (ddof=1) across three independent training runs, not across cycles. Positive improvement means matrix is better. Paired improvement is calculated separately per seed then averaged; ratio-of-mean-errors is also saved separately.

## Three-seed metrics

| metric | original mean +/- SD | matrix mean +/- SD | paired improvement % mean +/- SD |
| --- | --- | --- | --- |
| MAE | 0.006777026 +/- 0.00144 | 0.009958112 +/- 0.00296 | -55.196 +/- 73.257 |
| MAPE_percent | 0.7284093 +/- 0.1559 | 1.067587 +/- 0.3287 | -55.160 +/- 75.327 |
| RMSE | 0.008803757 +/- 0.001437 | 0.01766041 +/- 0.0101 | -110.811 +/- 139.768 |
| MSE | 7.888363e-05 +/- 2.65e-05 | 0.0003799181 +/- 0.0004146 | -474.646 +/- 688.696 |

![Three-seed comparison](wang_three_seed/three_seed_metrics.png)

## Individual runs

| arm | seed | MAE | MAPE_percent | RMSE | MSE | training_runtime_seconds | total_arm_seconds | best_epoch | stopping_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W-O | 42 | 0.006401181 | 0.6882775 | 0.008022624 | 6.43625e-05 | 439.9784 | 463.8086 | 111 | 200 |
| W-H | 42 | 0.00771128 | 0.8243499 | 0.01253254 | 0.0001570647 | 920.4663 | 972.0858 | 60 | 200 |
| W-O | 7 | 0.008368092 | 0.9004463 | 0.01046263 | 0.0001094666 | 358.3503 | 377.2454 | 113 | 200 |
| W-H | 7 | 0.008850348 | 0.9368656 | 0.01115124 | 0.0001243501 | 644.3324 | 680.1933 | 39 | 200 |
| W-O | 123 | 0.005561804 | 0.5965042 | 0.007926017 | 6.282174e-05 | 338.8296 | 356.9417 | 179 | 200 |
| W-H | 123 | 0.01331271 | 1.441545 | 0.02929743 | 0.0008583397 | 679.0813 | 717.8317 | 71 | 200 |

Runtime includes batch preparation/cache misses; validation is included in total-arm time and recorded separately in the CSV. It is measured, not inferred from the previous estimate.

## Battery-level consistency

| seed | batteries_improved | batteries_worsened | early_batteries_worsened | largest_excess_battery |
| --- | --- | --- | --- | --- |
| 42 | 21 | 15 | 34 | 2018-04-12_battery-33 |
| 7 | 22 | 14 | 36 | 2018-04-12_battery-33 |
| 123 | 19 | 17 | 18 | 2018-04-12_battery-33 |

Using each battery's mean MAE over the three seeds, 22/36 improve and 14/36 worsen. These are exploratory test-set diagnostics, not selection rules.

## Battery 33

| seed | battery33_original_MAE | battery33_matrix_MAE | battery33_improvement_percent | battery33_excess_rank | battery33_positive_excess_share_percent | battery33_matrix_squared_error_share_percent |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.002465833 | 0.02742021 | -1012.006 | 1 | 75.44967 | 61.51294 |
| 7 | 0.003755228 | 0.01742357 | -363.9817 | 1 | 36.54435 | 22.20425 |
| 123 | 0.004796137 | 0.07123999 | -1385.362 | 1 | 60.98704 | 83.02455 |

Excess absolute error is sample count times (matrix MAE - original MAE). Positive-excess share is relative to the sum over worsened batteries only; rank 1 is the largest excess contributor. All official metrics retain all 36 batteries.

## Early-cycle weakness

| seed | early_original_MAE | early_matrix_MAE | early_matrix_bias | early_batteries_worsened |
| --- | --- | --- | --- | --- |
| 42 | 0.003220022 | 0.009477642 | 0.007282614 | 34 |
| 7 | 0.003087937 | 0.01456846 | 0.01456846 | 36 |
| 123 | 0.005115958 | 0.005713387 | 0.00330364 | 18 |

Early means the first 20% of each recorded CSV trajectory, matching the previous detailed analysis. The full recorded length is used only for retrospective grouping, never as model input. Positive signed bias means overprediction. This is recorded position, not a verified physical-age or end-of-life definition.

![Battery 33 and position profiles](wang_three_seed/battery33_and_early_cycles.png)

## Training stability

| seed | arm | best_epoch | final_to_best_validation_ratio | postwarmup_validation_jumps_over_5x |
| --- | --- | --- | --- | --- |
| 42 | W-O | 111 | 1.269914 |  |
| 42 | W-H | 60 | 41.13162 | 61 |
| 7 | W-O | 113 | 1.38096 | 58 |
| 7 | W-H | 39 | 41.27262 | 46 |
| 123 | W-O | 179 | 1.102401 |  |
| 123 | W-H | 71 | 64.21823 | 77 |

The jump rule is descriptive: validation MSE greater than 5 times the previous epoch after epoch 30, matching the earlier audit. This does not identify a causal optimizer defect. Test metrics use the best validation epoch, not the final epoch.

![Validation by seed](wang_three_seed/validation_by_seed.png)

## Interpretation of the completed runs

The matrix pipeline has worse pooled MAE, MAPE, RMSE and MSE in all three paired seeds. The ratio of mean errors gives improvements of -46.94% (MAE), -46.56% (MAPE), -100.60% (RMSE) and -381.62% (MSE). These ratios differ from the mean of per-seed percentage improvements reported above; both are retained and explicitly labeled.

The result is not a uniform loss on every battery: 21, 22 and 19 of 36 batteries improve for seeds 42, 7 and 123 respectively, and 22 improve when comparing each battery's three-seed mean MAE. Nevertheless, severe errors recur in the same difficult battery. Battery 33 ranks first in excess absolute error in every seed, accounting for 75.45%, 36.54% and 60.99% of positive excess absolute error. Its matrix MAE is 0.02742, 0.01742 and 0.07124, versus original MAE 0.00247, 0.00376 and 0.00480. Seed 123 is particularly severe: this battery contributes 83.02% of matrix squared error. This is a reproducible failure concentration with seed-dependent severity, not merely a one-seed outlier.

Pooled early-cycle MAE is worse for matrix in all three seeds, with positive signed bias each time. However, the breadth varies: early MAE worsens on 34/36, 36/36 and 18/36 batteries. Therefore widespread early weakness reproduces strongly in seeds 42 and 7; seed 123 has a smaller pooled early penalty and only half the batteries worsen early. Do not describe every seed as having universal early failure.

Matrix validation degradation also repeats: the large post-warmup jumps occur at epochs 61, 46 and 77; final validation MSE is about 41, 41 and 64 times its best. Best matrix checkpoints are epochs 60, 39 and 71, all before the respective jump. Original seed 7 also has one threshold-crossing validation fluctuation at epoch 58, but recovers; the persistent matrix plateau is the distinguishing pattern. The later jumps cannot directly cause predictions from earlier saved checkpoints, and these logs do not identify a causal optimizer defect.

Conclusion: the current matrix pipeline is consistently worse on pooled test metrics and less stable under the fixed settings. It can help many batteries, so the evidence does not rule out a useful representation after controlled validation-led investigation. It does rule out claiming a robust improvement from the current implementation. No tuning, Zhang runs or FFT/no-FFT ablation was started.

The four additional runs consumed 33.68 minutes in training and 35.54 minutes in total arm time (including validation, evaluation and saving), excluding setup and report generation.

## Limitations and next steps

Three training seeds quantify initialization/order variability on one fixed held-out MIT date batch; they do not establish population significance. Matrix uses a larger first layer and 16-cycle history, so this is a pipeline comparison, not proof of an FFT-specific effect. No model tuning was performed. Any future tuning must use training/validation data, not these inspected test outcomes. Stop here; Zhang and ablations require a separate instruction.

## Artifacts

final_comparison.csv contains Wang aggregate means and sample SD only (two rows). wang_three_seed/ contains all six per-run metrics, 216 per-battery metric rows, paired improvements, per-seed diagnostics, aggregate battery MAE and plots. New checkpoints, histories and predictions are in W-O/seed_7, W-H/seed_7, W-O/seed_123 and W-H/seed_123. Source/configuration/manifests and diagnostic checks are in wang_shared_seed_7 and wang_shared_seed_123.

A setup-only variable-name error in the additional-seed runner was corrected before any training began; the failed setup directory is retained as wang_seed_7_setup_error. This did not change model, numerical settings or data protocol.
