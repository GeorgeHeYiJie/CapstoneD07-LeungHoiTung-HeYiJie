# MIT comparison: Wang three seeds and Zhang seed 42

Completed seeds 42, 7 and 123 for W-O and W-H. At the Wang-stage completion, no Zhang experiments or ablations had been run; the newly completed Zhang stage is reported below. Existing seed-42 outputs and the detailed single-seed report remain unchanged.

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

Conclusion: the current matrix pipeline is consistently worse on pooled test metrics and less stable under the fixed settings. It can help many batteries, so the evidence does not rule out a useful representation after controlled validation-led investigation. It does rule out claiming a robust improvement from the current implementation. No tuning or ablation was performed; the subsequent Zhang stage is reported below.

The four additional runs consumed 33.68 minutes in training and 35.54 minutes in total arm time (including validation, evaluation and saving), excluding setup and report generation.

## Limitations and next steps

Three training seeds quantify initialization/order variability on one fixed held-out MIT date batch; they do not establish population significance. Matrix uses a larger first layer and 16-cycle history, so this is a pipeline comparison, not proof of an FFT-specific effect. No model tuning was performed. Any future tuning must use training/validation data, not these inspected test outcomes. The subsequent authorized Zhang seed-42 stage is reported below; ablations remain unperformed.

## Artifacts

The previous two-row final_comparison.csv is preserved in zhang_shared_seed_42/previous_final_comparison.csv; the current table also includes Zhang. wang_three_seed/ contains all six per-run metrics, 216 per-battery metric rows, paired improvements, per-seed diagnostics, aggregate battery MAE and plots. New checkpoints, histories and predictions are in W-O/seed_7, W-H/seed_7, W-O/seed_123 and W-H/seed_123. Source/configuration/manifests and diagnostic checks are in wang_shared_seed_7 and wang_shared_seed_123.

A setup-only variable-name error in the additional-seed runner was corrected before any training began; the failed setup directory is retained as wang_seed_7_setup_error. This did not change model, numerical settings or data protocol.


## Zhang seed-42 comparison

Only Z-O and Z-H were newly trained, 200 epochs each with Adam lr=0.001 and no scheduler/early stopping. Existing model classes and losses are unchanged. These are BatLiNet-inspired SOH adaptations, not a full reproduction of the Zhang lifetime model.

Frozen HD-SOH-v1 matches Wang: 71/18/36 batteries, capacity/1.1 targets, train-only min-max, same 29,083 test keys and labels, same standard metric definitions. Input shapes are [B,1,16,16,1] original and [B,2,16,8,8] matrix. Parameter counts remain 9,313 and 18,529 respectively. Thus this is a pipeline comparison with differing input channels, not an isolated FFT effect.

References use training batteries only. Training reference RNG=10042; each evaluation resets its independent RNG (validation 20042, test 30042), fixing two distinct references per sample across epochs and arms. Reference IDs, clamped rows and clamp counts are saved in zhang_shared_seed_42/. Training batch order and reference schedule are paired across arms. Checkpoints are selected using final mixed-prediction validation MSE only.

| metric | original | matrix | improvement_percent |
| --- | --- | --- | --- |
| MAE | 0.01650476 | 0.01334346 | 19.15387 |
| MAPE_percent | 1.779178 | 1.422424 | 20.05163 |
| RMSE | 0.0200833 | 0.01760358 | 12.34716 |
| MSE | 0.0004033387 | 0.000309886 | 23.1698 |

Positive improvement means matrix has lower error. MAE/RMSE use SOH fractions, MSE squared fractions, MAPE percent. Single-seed SD is undefined (blank in final_comparison.csv), not zero. Wang still has three seeds, so variability estimates are not comparable across methods.

| arm | training_runtime_seconds | validation_runtime_seconds | total_arm_seconds | inference_cold_seconds | inference_warm_seconds | best_epoch | stopping_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Z-O | 865.9777 | 179.5425 | 1051.822 | 2.399704 | 1.944501 | 72 | 200 |
| Z-H | 1405.793 | 248.3099 | 1667.417 | 8.420381 | 2.579115 | 124 | 200 |

Inference timings include batch/reference preparation. The first test pass can include cache misses; a small diagnostic batch was already cached. Predictions and checkpoint reloads are verified exactly.

Battery MAE improves on 34/36 and worsens on 2/36. Largest excess-absolute-error contributor: 2018-04-12_battery-39. Worst matrix MAE battery: 2018-04-12_battery-39.

### Outlier concentration

| battery | n | MAE_original | MAE_matrix | MAE_difference | MAE_improvement_percent | early_MAE_original | early_MAE_matrix | matrix_bias | max_absolute_error_matrix | excess_absolute_error | positive_excess_share_percent | matrix_squared_error_share_percent | matrix_MAE_vs_median_battery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-04-12_battery-39 | 91 | 0.03376329 | 0.0447742 | 0.01101091 | -32.61209 | 0.01095771 | 0.01272409 | 0.0447742 | 0.9787193 | 1.001993 | 74.66049 | 21.05058 | 3.438907 |
| 2018-04-12_battery-38 | 1296 | 0.0247734 | 0.02503581 | 0.000262402 | -1.059209 | 0.02626329 | 0.02949844 | 0.02503581 | 0.04070503 | 0.340073 | 25.33951 | 9.346078 | 1.922889 |
| 2018-04-12_battery-11 | 123 | 0.02301252 | 0.02120483 | -0.001807691 | 7.855251 | 0.01735376 | 0.01931961 | 0.02120483 | 0.0508604 | -0.222346 | 0 | 0.7161904 | 1.628648 |
| 2018-04-12_battery-9 | 100 | 0.017458 | 0.01523317 | -0.00222483 | 12.7439 | 0.01241885 | 0.0137482 | 0.01523317 | 0.06007481 | -0.222483 | 0 | 0.3529141 | 1.169992 |
| 2018-04-12_battery-43 | 230 | 0.01514048 | 0.01355905 | -0.001581433 | 10.44506 | 0.01264159 | 0.01449552 | 0.01350976 | 0.05539781 | -0.3637295 | 0 | 0.6891042 | 1.04141 |

Battery 33 specifically:

| battery | n | MAE_original | MAE_matrix | MAE_difference | MAE_improvement_percent | early_MAE_original | early_MAE_matrix | matrix_bias | max_absolute_error_matrix | excess_absolute_error | positive_excess_share_percent | matrix_squared_error_share_percent | matrix_MAE_vs_median_battery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-04-12_battery-33 | 2155 | 0.007850575 | 0.005534247 | -0.002316327 | 29.5052 | 0.009009407 | 0.007913099 | 0.005418617 | 0.0182268 | -4.991686 | 0 | 1.034793 | 0.4250609 |

Positive excess share is measured among worsened batteries; squared-error share is relative to the entire matrix test set. More than five times median battery MAE is a descriptive outlier flag, not a universal definition of catastrophe. Flagged batteries: []. No battery is removed from official metrics.

![Per-battery MAE](zhang_analysis/per_battery_mae_difference.png)

![Outlier trajectories](zhang_analysis/outlier_trajectories.png)

### Early-cycle behavior

| arm | n | MAE | MAPE_percent | RMSE | MSE | bias |
| --- | --- | --- | --- | --- | --- | --- |
| Z-O | 5831 | 0.01598322 | 1.656388 | 0.01772225 | 0.0003140781 | 0.01597528 |
| Z-H | 5831 | 0.01716194 | 1.778145 | 0.01902878 | 0.0003620944 | 0.01714751 |

Early MAE worsens on 28/36 batteries. Early is the first 20% of recorded rows, used only retrospectively for analysis. Positive bias is overprediction. Full error profiles by recorded position and SOH, with sample/battery counts, are saved in error_profiles.csv.

![Early and SOH profiles](zhang_analysis/early_and_soh_profiles.png)

### Validation behavior

| arm | best_epoch | final_to_best_validation_ratio | postwarmup_jumps_over_5x | last20_validation_mean | last20_validation_std |
| --- | --- | --- | --- | --- | --- |
| Z-O | 72 | 1.113932 |  | 7.654196e-05 | 4.763666e-06 |
| Z-H | 124 | 1.014361 |  | 7.045051e-05 | 1.246908e-05 |

The >5x jump threshold after epoch 30 is the same descriptive rule used for Wang. A transient fluctuation alone does not establish persistent instability. Final/best ratios and the full curves must also be inspected. Fixed reference selection prevents epoch-to-epoch reference resampling from confounding validation.

![Training and validation](zhang_analysis/training_validation.png)

![Prediction versus truth](zhang_analysis/prediction_vs_true.png)

### Interpretation of Zhang seed 42

The matrix pipeline improves all four pooled metrics: MAE 19.15%, MAPE 20.05%, RMSE 12.35%, MSE 23.17%. Improvement is broad at the battery level (34/36), rather than caused by one lucky battery. Only batteries 38 and 39 worsen in full-trajectory MAE. This is encouraging for this seed, but does not establish multi-seed reliability or an FFT-specific benefit; input channels and parameter counts differ.

Battery 33 is not the Zhang failure case: its MAE falls from 0.007851 to 0.005534 (29.51% improvement), and it accounts for only 1.03% of matrix squared error. However, battery 39 has catastrophic late-sample overprediction. Its MAE rises from 0.033763 to 0.044774; it contributes 21.05% of total matrix squared error despite containing only 91/29,083 test samples. At original CSV row 90, true SOH is 0.802718, original predicts 1.322480 and matrix predicts 1.781437. Rows 89 and 90 alone contribute 18.52% of matrix squared error. A battery-average threshold (>5 times median battery MAE) misses this localized failure, so the empty flag list below must not be read as absence of catastrophic samples. Largest sample errors are saved separately in largest_sample_errors.csv.

Inspection of the existing features shows a sharp distribution shift in battery 39: the largest absolute train-normalized feature increases from about 1.18 at row 87 to 33.53, 66.44 and 65.80 at rows 88-90. This coincides with the prediction failures in both representations, but does not establish causation. The frozen protocol intentionally does not clip test inputs or remove outliers; no rows were removed and no model was changed.

Early-cycle weakness remains: pooled first-20%-trajectory MAE rises from 0.015983 to 0.017162 (7.37% worse), and 28/36 batteries worsen early. Both models overpredict in this region. Matrix gains are mainly in the middle/later recorded trajectory and lower-SOH bands; see the binned profiles rather than assuming uniform improvement.

Training does not reproduce Wang matrix's persistent collapse in this run. Neither Zhang arm has a post-epoch-30 validation jump exceeding 5 times its previous value; smaller transient spikes occur, especially for matrix, and recover. Final/best validation MSE ratios are 1.114 (original) and 1.014 (matrix), versus about 41-64 for the Wang matrix runs. This is evidence of stable completion for seed 42, not proof of stability across seeds.

Best epochs are 72 (original) and 124 (matrix). Measured total-arm runtimes are 17.53 and 27.79 minutes, totaling 45.32 minutes; training alone totals 37.86 minutes. Setup and analysis time are outside those totals. No Wang training, further Zhang seeds, tuning or ablations were run.

Reference-row clamping is part of the frozen protocol: 9,144/27,170 validation reference assignments and 24,640/58,166 test reference assignments use the last available training-reference row. The assignments are identical across the two arms and do not consult evaluation targets, but row-index alignment is an approximation rather than physical-age matching.

Conclusion: Zhang matrix is promising in this one-seed pipeline comparison, with broad per-battery improvement and no persistent training collapse, while retaining early-SOH overprediction and a severe battery-39 tail failure. It is not yet a robust multi-seed result and does not show that FFT alone caused the gain.

### Artifacts and scope

New checkpoints, full histories and predictions: Z-O/seed_42 and Z-H/seed_42. Shared configuration/source snapshots/manifests/smoke checks: zhang_shared_seed_42. Per-run metrics, all 72 battery metric rows, paired improvements and plots: zhang_analysis. final_comparison.csv now contains Wang three-seed aggregates and Zhang one-seed values. No Wang training, architecture changes, hyperparameter tuning or ablation was performed. Stop after these two runs.
