# TEST2: OFFLINE original versus denoised MIT results

**Offline only.** Centered smoothing and full-trajectory pooling/interpolation use future-cycle information. These results do not establish real-time deployment performance.

One seed (42), 200 epochs per condition, no tuning or early stopping. The same 71/18/36 train/validation/test batteries and 29,083 held-out test samples are used. Each predictor is selected solely by its lowest full-validation SOH MSE. Labels and row IDs match exactly.

Train-only scalers are fitted separately using the same procedure for each feature condition. No outlier filtering or clipping changes the evaluated rows. SOH is capacity/1.1 in fraction units. MAE/RMSE use fractional SOH; MSE uses its square; MAPE uses true SOH as denominator and is a percentage. Overall metrics are sample-weighted.

The comparison is within this test; the other test is not its baseline. All smoke results are excluded.

## combined: numerical comparison

| metric | original | denoised | denoised_minus_original | absolute_difference | improvement_percent |
| --- | --- | --- | --- | --- | --- |
| MAE | 0.02035829 | 0.03528502 | 0.01492673 | 0.01492673 | -73.32013 |
| MAPE_percent | 2.172563 | 3.697931 | 1.525368 | 1.525368 | -70.21053 |
| RMSE | 0.03403092 | 0.06504626 | 0.03101534 | 0.03101534 | -91.1387 |
| MSE | 0.001158104 | 0.004231016 | 0.003072912 | 0.003072912 | -265.34 |

Improvement = (original error − denoised error) / original error × 100. Positive is better; negative is worse. Signed difference is denoised minus original; absolute_difference is its magnitude.

Battery MAE: **9/36 improved; 27/36 worsened**. Macro-average battery MAE: 0.023495388 original versus 0.037556453 denoised.
The three largest positive excess-absolute-error contributors account for 45.27% of total positive excess error among worsened batteries. This measures concentration of worsening, not net overall improvement.
The three largest absolute-error reductions account for 52.96% of total positive error reductions among improved batteries. Median battery MAE improvement is -57.43%. Improvement is not a majority-battery result; more than half of the positive gain is concentrated in three batteries. These are descriptive concentration measures, not significance tests.

Largest MAE improvements:

| battery_id | MAE_original | MAE_denoised | MAE_improvement_percent |
| --- | --- | --- | --- |
| 2018-04-12_battery-40 | 0.01308129 | 0.002794154 | 78.64008 |
| 2018-04-12_battery-34 | 0.01528492 | 0.00422273 | 72.37322 |
| 2018-04-12_battery-19 | 0.01380045 | 0.00406464 | 70.54706 |
| 2018-04-12_battery-22 | 0.01297649 | 0.005719547 | 55.92379 |
| 2018-04-12_battery-39 | 0.08405687 | 0.04796937 | 42.93225 |
| 2018-04-12_battery-37 | 0.007217774 | 0.004319479 | 40.15497 |
| 2018-04-12_battery-36 | 0.01846183 | 0.01202422 | 34.86985 |
| 2018-04-12_battery-38 | 0.01335327 | 0.009825057 | 26.42209 |
| 2018-04-12_battery-18 | 0.01653682 | 0.0137329 | 16.95566 |
| 2018-04-12_battery-6 | 0.009738518 | 0.01131628 | -16.2013 |

Worst MAE changes:

| battery_id | MAE_original | MAE_denoised | MAE_improvement_percent |
| --- | --- | --- | --- |
| 2018-04-12_battery-24 | 0.0121414 | 0.05988815 | -393.2558 |
| 2018-04-12_battery-4 | 0.007094023 | 0.03035353 | -327.8747 |
| 2018-04-12_battery-27 | 0.02206551 | 0.08479553 | -284.29 |
| 2018-04-12_battery-2 | 0.006338335 | 0.02295908 | -262.2257 |
| 2018-04-12_battery-31 | 0.02765756 | 0.07471328 | -170.1369 |
| 2018-04-12_battery-1 | 0.006803428 | 0.01822328 | -167.8543 |
| 2018-04-12_battery-43 | 0.0266985 | 0.06393476 | -139.4695 |
| 2018-04-12_battery-8 | 0.02446578 | 0.05368078 | -119.4117 |
| 2018-04-12_battery-42 | 0.02950132 | 0.06390833 | -116.6287 |
| 2018-04-12_battery-26 | 0.02785634 | 0.05724865 | -105.5139 |

Early-cycle pooled MAE: 0.019188028 original versus 0.10423235 denoised (-443.22% improvement). Early MAE worsened on 35/36 batteries.
Early signed bias: +0.013131396 original versus -0.10287859 denoised. Positive bias means overprediction.

Training/validation stability:

| predictor | condition | best_epoch | best_validation_MSE | final_validation_MSE | final_to_best_ratio | jumps_over_5x_after_epoch30 | max_validation_MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combined | original | 76 | 3.343022e-05 | 4.827397e-05 | 1.444022 |  | 0.0012731 |
| combined | denoised | 118 | 5.137773e-05 | 7.82865e-05 | 1.523744 |  | 0.06215977 |

Runtime and selection:

| condition | seed | training_seconds | validation_seconds | test_inference_seconds | total_run_seconds | best_epoch | stopping_epoch | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 42 | 21254.14 | 101.1031 | 1.05869 | 21359.33 | 76 | 200 | combined_best.pt |
| denoised | 42 | 19281.05 | 91.70312 | 1.323437 | 19376.97 | 118 | 200 | combined_best.pt |

Largest individual denoised prediction errors:

| predictor | condition | battery_id | MAE | max_absolute_error | worst_row | true_SOH | predicted_SOH | outside_0_1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined | denoised | 2018-04-12_battery-39 | 0.04796937 | 2.306643 | 90 | 0.8027175 | -1.503926 | 2 |
| combined | denoised | 2018-04-12_battery-27 | 0.08479553 | 0.3625983 | 32 | 0.9778391 | 0.6152408 | 0 |
| combined | denoised | 2018-04-12_battery-31 | 0.07471328 | 0.2838874 | 17 | 0.9740635 | 0.6901761 | 0 |
| combined | denoised | 2018-04-12_battery-45 | 0.05246099 | 0.2774943 | 21 | 0.9756703 | 0.698176 | 0 |
| combined | denoised | 2018-04-12_battery-24 | 0.05988815 | 0.2599219 | 35 | 0.9710492 | 0.7111273 | 0 |

Cautious finding: denoising increased pooled MAE by 73.32% in this seed. The battery counts, macro-average and outlier concentration above determine whether this reflects broad or concentrated changes. This is not a statistical-significance claim and does not establish an online benefit.

Runtime for Test 1 is the shared Wang+Zhang condition runtime, not a separately timed training cost for each predictor. Matrix-cache preparation is recorded under prepared/ and excluded from training runtime. Validation stability alone does not establish test robustness. Any future tuning must use training/validation data, not the inspected test outcomes.

See comparison_metrics.csv, per_battery_metrics.csv, early_cycle_metrics.csv, outlier_metrics.csv, stability.csv and runtime.csv for complete numerical results. Predictions/checkpoints and full training-loss histories remain under original/ and denoised/.
