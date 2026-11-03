# TEST1: OFFLINE original versus denoised MIT results

**Offline only.** Centered smoothing and full-trajectory pooling/interpolation use future-cycle information. These results do not establish real-time deployment performance.

One seed (42), 200 epochs per condition, no tuning or early stopping. The same 71/18/36 train/validation/test batteries and 29,083 held-out test samples are used. Each predictor is selected solely by its lowest full-validation SOH MSE. Labels and row IDs match exactly.

Train-only scalers are fitted separately using the same procedure for each feature condition. No outlier filtering or clipping changes the evaluated rows. SOH is capacity/1.1 in fraction units. MAE/RMSE use fractional SOH; MSE uses its square; MAPE uses true SOH as denominator and is a percentage. Overall metrics are sample-weighted.

The comparison is within this test; the other test is not its baseline. All smoke results are excluded.

## wang: numerical comparison

| metric | original | denoised | denoised_minus_original | absolute_difference | improvement_percent |
| --- | --- | --- | --- | --- | --- |
| MAE | 0.005294443 | 0.007500485 | 0.002206042 | 0.002206042 | -41.66712 |
| MAPE_percent | 0.5699505 | 0.812992 | 0.2430415 | 0.2430415 | -42.64256 |
| RMSE | 0.007272811 | 0.01048526 | 0.003212445 | 0.003212445 | -44.17061 |
| MSE | 5.289378e-05 | 0.0001099406 | 5.704682e-05 | 5.704682e-05 | -107.8517 |

Improvement = (original error − denoised error) / original error × 100. Positive is better; negative is worse. Signed difference is denoised minus original; absolute_difference is its magnitude.

Battery MAE: **8/36 improved; 28/36 worsened**. Macro-average battery MAE: 0.0052482169 original versus 0.0076518346 denoised.
The three largest positive excess-absolute-error contributors account for 25.89% of total positive excess error among worsened batteries. This measures concentration of worsening, not net overall improvement.
The three largest absolute-error reductions account for 87.38% of total positive error reductions among improved batteries. Median battery MAE improvement is -68.65%. Improvement is not a majority-battery result; more than half of the positive gain is concentrated in three batteries. These are descriptive concentration measures, not significance tests.

Largest MAE improvements:

| battery_id | MAE_original | MAE_denoised | MAE_improvement_percent |
| --- | --- | --- | --- |
| 2018-04-12_battery-9 | 0.008445436 | 0.005087906 | 39.75556 |
| 2018-04-12_battery-37 | 0.005184399 | 0.003320559 | 35.95092 |
| 2018-04-12_battery-33 | 0.01152357 | 0.008255873 | 28.35663 |
| 2018-04-12_battery-22 | 0.006972351 | 0.005252642 | 24.66469 |
| 2018-04-12_battery-11 | 0.008525984 | 0.006684251 | 21.60141 |
| 2018-04-12_battery-6 | 0.005354792 | 0.004843326 | 9.551547 |
| 2018-04-12_battery-30 | 0.006062834 | 0.005693578 | 6.090485 |
| 2018-04-12_battery-24 | 0.005081166 | 0.005001564 | 1.566611 |
| 2018-04-12_battery-14 | 0.005486993 | 0.005750697 | -4.80598 |
| 2018-04-12_battery-26 | 0.006135811 | 0.006787036 | -10.61351 |

Worst MAE changes:

| battery_id | MAE_original | MAE_denoised | MAE_improvement_percent |
| --- | --- | --- | --- |
| 2018-04-12_battery-45 | 0.002965357 | 0.008230342 | -177.5498 |
| 2018-04-12_battery-39 | 0.008134897 | 0.02065445 | -153.8993 |
| 2018-04-12_battery-44 | 0.0035992 | 0.008563641 | -137.9317 |
| 2018-04-12_battery-21 | 0.002986509 | 0.006929548 | -132.0284 |
| 2018-04-12_battery-36 | 0.003692432 | 0.008459876 | -129.1139 |
| 2018-04-12_battery-15 | 0.003998834 | 0.00852626 | -113.2187 |
| 2018-04-12_battery-27 | 0.003574569 | 0.007362967 | -105.982 |
| 2018-04-12_battery-10 | 0.00503883 | 0.01025032 | -103.4266 |
| 2018-04-12_battery-31 | 0.003588508 | 0.007140067 | -98.97032 |
| 2018-04-12_battery-1 | 0.003569558 | 0.00677003 | -89.66018 |

Early-cycle pooled MAE: 0.0049260909 original versus 0.0028293662 denoised (+42.56% improvement). Early MAE worsened on 8/36 batteries.
Early signed bias: +0.0015409354 original versus -0.00010881238 denoised. Positive bias means overprediction.

Training/validation stability:

| predictor | condition | best_epoch | best_validation_MSE | final_validation_MSE | final_to_best_ratio | jumps_over_5x_after_epoch30 | max_validation_MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wang | original | 9 | 5.108348e-05 | 0.001358159 | 26.58706 |  | 0.001848525 |
| wang | denoised | 12 | 3.780522e-05 | 0.001358488 | 35.93386 |  | 0.002685106 |

Runtime and selection:

| condition | seed | training_seconds | validation_seconds | test_inference_seconds | total_run_seconds | best_epoch | stopping_epoch | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 42 | 3715.704 | 311.0332 | 4.152913 | 4036.385 | 9 | 200 | wang_best.pt |
| denoised | 42 | 4234.92 | 344.0504 | 3.926005 | 4588.566 | 12 | 200 | wang_best.pt |

Largest individual denoised prediction errors:

| predictor | condition | battery_id | MAE | max_absolute_error | worst_row | true_SOH | predicted_SOH | outside_0_1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wang | denoised | 2018-04-12_battery-39 | 0.02065445 | 0.1516781 | 90 | 0.8027175 | 0.9543957 | 0 |
| wang | denoised | 2018-04-12_battery-38 | 0.0113611 | 0.1062919 | 1295 | 0.8338588 | 0.9401507 | 0 |
| wang | denoised | 2018-04-12_battery-34 | 0.006168411 | 0.05486029 | 1208 | 0.8452122 | 0.9000725 | 0 |
| wang | denoised | 2018-04-12_battery-29 | 0.01013578 | 0.05191749 | 501 | 0.8237998 | 0.8757173 | 0 |
| wang | denoised | 2018-04-12_battery-46 | 0.01007617 | 0.05133981 | 1646 | 0.8484533 | 0.8997931 | 0 |

Cautious finding: denoising increased pooled MAE by 41.67% in this seed. The battery counts, macro-average and outlier concentration above determine whether this reflects broad or concentrated changes. This is not a statistical-significance claim and does not establish an online benefit.

## zhang: numerical comparison

| metric | original | denoised | denoised_minus_original | absolute_difference | improvement_percent |
| --- | --- | --- | --- | --- | --- |
| MAE | 0.01223739 | 0.01195869 | -0.000278697 | 0.000278697 | 2.277422 |
| MAPE_percent | 1.300948 | 1.276757 | -0.02419171 | 0.02419171 | 1.859544 |
| RMSE | 0.01980678 | 0.01447502 | -0.005331759 | 0.005331759 | 26.91886 |
| MSE | 0.0003923085 | 0.0002095262 | -0.0001827823 | 0.0001827823 | 46.59147 |

Improvement = (original error − denoised error) / original error × 100. Positive is better; negative is worse. Signed difference is denoised minus original; absolute_difference is its magnitude.

Battery MAE: **23/36 improved; 13/36 worsened**. Macro-average battery MAE: 0.014241069 original versus 0.012441085 denoised.
The three largest positive excess-absolute-error contributors account for 60.77% of total positive excess error among worsened batteries. This measures concentration of worsening, not net overall improvement.
The three largest absolute-error reductions account for 27.55% of total positive error reductions among improved batteries. Median battery MAE improvement is +8.23%. A majority of batteries improve; the three largest contributors account for no more than half of the positive gain. These are descriptive concentration measures, not significance tests.

Largest MAE improvements:

| battery_id | MAE_original | MAE_denoised | MAE_improvement_percent |
| --- | --- | --- | --- |
| 2018-04-12_battery-39 | 0.05429 | 0.02324582 | 57.18214 |
| 2018-04-12_battery-45 | 0.008837501 | 0.004860917 | 44.9967 |
| 2018-04-12_battery-44 | 0.01183034 | 0.006783216 | 42.66254 |
| 2018-04-12_battery-16 | 0.010528 | 0.006064772 | 42.39386 |
| 2018-04-12_battery-10 | 0.008931054 | 0.005864323 | 34.33784 |
| 2018-04-12_battery-21 | 0.01271595 | 0.008684941 | 31.70042 |
| 2018-04-12_battery-42 | 0.02367947 | 0.01683792 | 28.89233 |
| 2018-04-12_battery-12 | 0.009745478 | 0.006940371 | 28.78368 |
| 2018-04-12_battery-31 | 0.01087182 | 0.00781642 | 28.10383 |
| 2018-04-12_battery-9 | 0.01565768 | 0.01232692 | 21.27238 |

Worst MAE changes:

| battery_id | MAE_original | MAE_denoised | MAE_improvement_percent |
| --- | --- | --- | --- |
| 2018-04-12_battery-18 | 0.00644051 | 0.01014515 | -57.52095 |
| 2018-04-12_battery-19 | 0.007175434 | 0.01057457 | -47.37189 |
| 2018-04-12_battery-38 | 0.02201768 | 0.02746047 | -24.72009 |
| 2018-04-12_battery-46 | 0.01207869 | 0.01492182 | -23.53843 |
| 2018-04-12_battery-24 | 0.01017461 | 0.0117878 | -15.85505 |
| 2018-04-12_battery-14 | 0.00930135 | 0.01024333 | -10.12734 |
| 2018-04-12_battery-36 | 0.01102086 | 0.01210138 | -9.804326 |
| 2018-04-12_battery-30 | 0.007166004 | 0.007793835 | -8.761251 |
| 2018-04-12_battery-40 | 0.02013284 | 0.02183418 | -8.450593 |
| 2018-04-12_battery-33 | 0.007691003 | 0.008331497 | -8.327839 |

Early-cycle pooled MAE: 0.015079006 original versus 0.012596557 denoised (+16.46% improvement). Early MAE worsened on 5/36 batteries.
Early signed bias: +0.014861446 original versus +0.012244622 denoised. Positive bias means overprediction.

Training/validation stability:

| predictor | condition | best_epoch | best_validation_MSE | final_validation_MSE | final_to_best_ratio | jumps_over_5x_after_epoch30 | max_validation_MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zhang | original | 108 | 5.935072e-05 | 6.486732e-05 | 1.092949 |  | 0.0002639906 |
| zhang | denoised | 73 | 5.29807e-05 | 6.312176e-05 | 1.19141 |  | 0.0005764143 |

Runtime and selection:

| condition | seed | training_seconds | validation_seconds | test_inference_seconds | total_run_seconds | best_epoch | stopping_epoch | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 42 | 3715.704 | 311.0332 | 4.152913 | 4036.423 | 108 | 200 | zhang_best.pt |
| denoised | 42 | 4234.92 | 344.0504 | 3.926005 | 4588.601 | 73 | 200 | zhang_best.pt |

Largest individual denoised prediction errors:

| predictor | condition | battery_id | MAE | max_absolute_error | worst_row | true_SOH | predicted_SOH | outside_0_1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zhang | denoised | 2018-04-12_battery-39 | 0.02324582 | 0.6734506 | 90 | 0.8027175 | 0.1292669 | 0 |
| zhang | denoised | 2018-04-12_battery-46 | 0.01492182 | 0.04932141 | 1630 | 0.8538097 | 0.9031311 | 0 |
| zhang | denoised | 2018-04-12_battery-11 | 0.01860731 | 0.04460514 | 116 | 0.8642086 | 0.9088137 | 0 |
| zhang | denoised | 2018-04-12_battery-38 | 0.02746047 | 0.04431605 | 920 | 0.9054294 | 0.9497454 | 0 |
| zhang | denoised | 2018-04-12_battery-26 | 0.01621331 | 0.04300618 | 124 | 0.8557451 | 0.8987513 | 0 |

Cautious finding: denoising reduced pooled MAE by 2.28% in this seed. The battery counts, macro-average and outlier concentration above determine whether this reflects broad or concentrated changes. This is not a statistical-significance claim and does not establish an online benefit.

Runtime for Test 1 is the shared Wang+Zhang condition runtime, not a separately timed training cost for each predictor. Matrix-cache preparation is recorded under prepared/ and excluded from training runtime. Validation stability alone does not establish test robustness. Any future tuning must use training/validation data, not the inspected test outcomes.

See comparison_metrics.csv, per_battery_metrics.csv, early_cycle_metrics.csv, outlier_metrics.csv, stability.csv and runtime.csv for complete numerical results. Predictions/checkpoints and full training-loss histories remain under original/ and denoised/.
