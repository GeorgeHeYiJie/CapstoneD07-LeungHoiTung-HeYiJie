# Offline denoising comparison: completed results

**OFFLINE ONLY:** the denoiser uses future-cycle information. These results do not support a real-time deployment claim.

Seed 42; 200 epochs per condition; 71/18/36 frozen MIT batteries; 29,083 matching test samples; condition-specific training-only normalization; no outlier filtering; independent validation-MSE checkpoint selection.

The comparisons below are original versus denoised within each test/predictor, not Test 1 versus Test 2. Positive improvement means lower denoised error.

| predictor | metric | original | denoised | denoised_minus_original | absolute_difference | improvement_percent | test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wang | MAE | 0.005294443 | 0.007500485 | 0.002206042 | 0.002206042 | -41.66712 | test1 |
| wang | MAPE_percent | 0.5699505 | 0.812992 | 0.2430415 | 0.2430415 | -42.64256 | test1 |
| wang | RMSE | 0.007272811 | 0.01048526 | 0.003212445 | 0.003212445 | -44.17061 | test1 |
| wang | MSE | 5.289378e-05 | 0.0001099406 | 5.704682e-05 | 5.704682e-05 | -107.8517 | test1 |
| zhang | MAE | 0.01223739 | 0.01195869 | -0.000278697 | 0.000278697 | 2.277422 | test1 |
| zhang | MAPE_percent | 1.300948 | 1.276757 | -0.02419171 | 0.02419171 | 1.859544 | test1 |
| zhang | RMSE | 0.01980678 | 0.01447502 | -0.005331759 | 0.005331759 | 26.91886 | test1 |
| zhang | MSE | 0.0003923085 | 0.0002095262 | -0.0001827823 | 0.0001827823 | 46.59147 | test1 |
| combined | MAE | 0.02035829 | 0.03528502 | 0.01492673 | 0.01492673 | -73.32013 | test2 |
| combined | MAPE_percent | 2.172563 | 3.697931 | 1.525368 | 1.525368 | -70.21053 | test2 |
| combined | RMSE | 0.03403092 | 0.06504626 | 0.03101534 | 0.03101534 | -91.1387 | test2 |
| combined | MSE | 0.001158104 | 0.004231016 | 0.003072912 | 0.003072912 | -265.34 | test2 |

Detailed battery, early-cycle, outlier, stability, runtime and checkpoint findings:

- [Test 1 result analysis](test1/RESULT_ANALYSIS.md)
- [Test 2 result analysis](test2/RESULT_ANALYSIS.md)

Each test retains its prediction arrays, best checkpoints, training histories and verification records. One seed and one held-out MIT batch do not establish statistical significance or generalization. No tuning was performed.
