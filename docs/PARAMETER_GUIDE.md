# Denoising and combined models: parameter roles and interactions

Updated: 3 November 2026. This English guide replaces the Chinese guide. No model code, experiment settings or results were changed, and no training was run.

The original guide was prepared after synchronizing with shared `group/main` at `874a69f2bd45e07f796f39cb0167afe272d2417c`. The completed experiments used source commit `8952464066c66844e624ae5c3d2142296e41b7ba`; the three root algorithm scripts were unchanged between these revisions. The subsequently added `运行用的Oct29/` folder provides another execution entry point; it does not mean the completed CPU experiments used a GPU.

**This guide explains the code, not the results of parameter tuning. No single-parameter ablation was performed, so observed error changes cannot be attributed to an individual parameter.**

## 1. Roles of the scripts

| File | Role in these experiments |
| --- | --- |
| [DenoisingAlgorithm.py](../DenoisingAlgorithm.py) | Denoises each battery's 16 input features; leaves capacity unchanged |
| [HighDimensionMatrix-PINN_Combination.py](../HighDimensionMatrix-PINN_Combination.py) | Test 1: trains and evaluates the Wang and Zhang-inspired predictors separately; their outputs are not fused into a single SOH prediction |
| [New1HighDimensionMatrix-PINN_Combination.py](../New1HighDimensionMatrix-PINN_Combination.py) | Test 2: combines the Wang and Zhang-inspired branches into one weighted SOH prediction |
| [run_denoising_analysis.py](../scripts/run_denoising_analysis.py) | Implements the approved frozen evaluation protocol, including sample selection, training settings, logging and checkpoint selection, without editing the original model classes |

The current Test 2 training does not supply `feat_ref`, so `zhang.head_delta` does not participate in training or prediction. This implementation should not be described as a complete implementation of all cross-battery learning methods from both papers.

## 2. Denoising parameters

These values come from the constants and functions in `DenoisingAlgorithm.py` and were used to generate the experiment's denoised data.

| Parameter | Value | Meaning and interactions |
| --- | --- | --- |
| `DATA_DIR` / `OUTPUT_DIR` | Placeholder input / `./denoised_for_matrix` in the original script | File paths, not model hyperparameters. The experiment adapter processes each date directory and saves to `prepared/denoised_csv/`, without overwriting source data |
| `FEATURE_COLS` | 16 columns | Selects the features to process; column names must match the model inputs |
| `CAPACITY_COL` | `capacity` | Preserves the original capacity values; SOH labels are not denoised |
| `SG_WINDOW` | 11 | Savitzky–Golay window spanning consecutive rows. A larger window generally favors smoothing but may suppress short-term features; an optimal value has not been established experimentally |
| `SG_POLY` | 2 | Fits a quadratic polynomial within the window. Must be smaller than the window length; together, these settings control local curve flexibility |
| `GCL_SIGMA` | 0.08 | Scale of residual weighting. Increasing sigma generally makes the weight decay more slowly with residual magnitude |
| `GCL_ALPHA` | 1.5 | Controls the shape of the residual-weighting curve; unrelated to the PINN PDE loss weight |
| `GCL_ITER` | 3 | Number of residual-correction iterations. More iterations do not necessarily mean more smoothing because each update adds part of the original residual back |
| Correction coefficient inside the function | 0.4 | In `filtered += 0.4 * w * residual`, controls how much weighted residual is added back at each iteration |
| `SCALES` | `[2,4,8,16]` | Averages consecutive blocks at different sizes, then interpolates each result back to the original length |
| `FUSION_MODE` | `weighted` | Uses weighted fusion; the alternative `average` gives equal weight to each scale |
| `SCALE_WEIGHTS` | `[0.4,0.3,0.2,0.1]` | Corresponds to scales 2, 4, 8 and 16, respectively. The code normalizes the weights. Emphasizing larger scales favors their coarser trends, but does not guarantee better prediction |

Processing order: **original features → SG smoothing → GCL residual correction → multi-scale averaging/interpolation → weighted fusion → training-data-based normalization → high-dimensional matrices**.

Denoising occurs **before normalization**. The 16 columns have different units and numerical ranges but share `GCL_SIGMA=0.08`. Consequently, this value does not represent the same relative correction scale for every feature. This is a possible factor to investigate, not a demonstrated cause of prediction errors.

The multi-scale function uses `L // scale` complete blocks, where L is the trajectory length. A trailing incomplete block does not directly enter the block averages. The function then interpolates back to L rows using evenly spaced coordinates over the full length. Preserving row count therefore does not mean the original temporal positions are unaffected.

**OFFLINE limitation:** centered smoothing and full-trajectory pooling/interpolation use future-cycle information. A smoother curve is not necessarily closer to an unknown true signal and does not necessarily produce more accurate SOH predictions.

## 3. High-dimensional matrices and inputs

| Setting | Test 1 | Test 2 | Meaning and interactions |
| --- | --- | --- | --- |
| Original features | 16 | 16 | Pre-extracted charging features per row, not raw voltage waveforms |
| `xt` | 17 dimensions | 17 dimensions | 16 normalized features plus a normalized row-based time coordinate |
| `WINDOW` / `--window` | 16 | 16 | Historical window for matrix construction; distinct from the denoising `SG_WINDOW=11` |
| `K_PERIODS` / `--k` | 2 | 3 | Number of retained period representations; changing it affects the representation and some network input dimensions |
| Matrix height and width | 8 × 8 | 8 × 8 | Resizes period matrices to a fixed shape for model processing |
| Matrix shape per sample | `[2,16,8,8]` | `[3,16,8,8]` | Periods, features, height and width; prepend B for a batch of B samples |

The matrix reorganizes existing feature history; it does not add new sensor measurements. The scripts also use different period-selection logic, so equal window lengths do not imply identical representations. Within each test, original and denoised conditions retain the same settings: the main comparison is the model's data pipeline with versus without denoising.

## 4. Loss weights and mixing weights

### Test 1: separate predictors

Wang: `L = L_data + 0.5 * L_PDE + 0.01 * L_trend`.

- `alpha=0.5` weights the PDE residual loss.
- `beta=0.01` weights the penalty for a predicted change whose direction conflicts with the observed label change.
- `L_trend = sum(ReLU((u2-u1)*(y1-y2)))`, where u denotes predictions, y denotes labels, and 1 and 2 denote adjacent rows.
- Data and PDE losses use mean squared errors, whereas trend loss uses a sum. Changing batch size can therefore change their relative scales. The value of beta alone does not determine constraint strength.

Zhang: `L = L_intra + 1.0 * L_delta`.

- `lambda_delta=1` weights the loss for learning SOH differences between batteries.
- `alpha_mix=0.5` gives `0.5*y_intra + 0.5*(y_delta+y_ref)`.
- Here, the mixture combines **Zhang's single-battery prediction and its reference-battery difference prediction**, not Wang and Zhang predictions.
- `n_refs=2` averages predictions using two fixed training reference batteries during validation/test. Reference labels come from training batteries, not held-out test batteries.

### Test 2: fused prediction

`y_hat = sigmoid(mix)*u_Wang + (1-sigmoid(mix))*y_Zhang`.

| Parameter | Value | Meaning |
| --- | --- | --- |
| `MIX_INIT` | 0.5 | Initial value of the learnable parameter mix. After sigmoid, the initial Wang weight is approximately 62.25% and the Zhang weight 37.75%, not 50/50 |
| `ALPHA_PHYS` | 0.1 | PDE loss weight; distinct from the output mixing weight |
| `BETA_MONO` | 0.01 | Weight of the Wang branch's adjacent-row trend penalty |
| Matrix correction coefficient | 0.1 | `u_Wang = main_path_output + 0.1*matrix_branch_output`; distinct from the PDE weight despite having the same numerical value |
| `HIDDEN` | 60 | Hidden width of the Wang encoder; changing it changes model capacity/architecture |
| Dropout | 0.2 | Randomly masks part of the representation during training; disabled during evaluation and affected by the random seed |

Test 2: `L = MSE(y_hat,y) + 0.1*mean(f²) + 0.01*L_trend`, where f is the Wang branch's PDE residual.

The data loss constrains the fused prediction. The PDE and trend terms act on the Wang branch; in particular, the trend term uses Wang's u rather than the fused prediction y_hat. The mixing parameter and branches interact during training, so the final mixing weight alone cannot establish a branch's independent causal contribution.

## 5. Actual formal-run settings, not source entry-point defaults

| Setting | Test 1 | Test 2 |
| --- | --- | --- |
| Model seed | 42 | 42 |
| Epochs | 200, no early stopping | 200, no early stopping |
| Optimizer | Three Adam optimizers: Wang solution, dynamics and Zhang | One Adam optimizer for the combined model |
| Learning rate | Solution: 30-epoch warmup from 0.002 toward 0.01, then cosine toward 0.0002; dynamics/Zhang: 0.001 | Fixed 0.001; no scheduler |
| Update unit | 64 adjacent-row pairs per batch | One adjacent-row pair, or 2 rows per update |
| Training stride | 1 | 2, non-overlapping adjacent pairs |
| Best checkpoint | Lowest full-validation SOH MSE, selected independently for Wang and Zhang | Lowest full-validation SOH MSE of the fused prediction |
| CPU threads | 1 | 1 |

Shared protocol: 71/18/36 MIT train/validation/test batteries; 29,083 test samples; SOH = capacity / 1.1 Ah; matching battery order, row IDs, labels and evaluation samples across conditions; no repeated outlier filtering. The split seed is 420, while the model seed is 42: these serve different purposes. Reference-sampling seeds are 10042 for training, 20042 for validation and 30042 for testing; Test 2 does not activate its reference-battery branch.

The two conditions use **the same training-only normalization procedure but separately fitted scaler values**. This evaluates the full denoising-plus-scaler pipeline, not an ablation holding the scaler coefficients identical.

The original Test 1 entry point defaults to 2 epochs, a small battery subset and stride 5; Test 2 defaults to 50 epochs. Both default to seed 420. Running those entry points directly will not reproduce this formal protocol. See [PROTOCOL.md](../results/denoising_analysis/PROTOCOL.md) for approved deviations and runtime fixes, and each run's `configuration.json` for recorded settings.

## 6. Existing documentation and limits of interpretation

- [Original README](../README.md): installation, data and execution instructions, not a complete parameter-interaction guide.
- Test 1's opening documentation and `argparse` help: relatively detailed workflows, equations and parameter descriptions.
- Denoising/Test 2 parameter sections: brief Chinese comments; fixed coefficients inside functions and active/inactive branches require inspecting the implementation.
- [PINN explanation](PINN_EXPLAINED.md) and [student version](PINN_EXPLAINED_FOR_STUDENT.md): original model and loss explanations.
- [Loss-weight discrepancy](LOSS_WEIGHT_DISCREPANCY.md): historical versus current formulas and weights. Numerical weights are not directly equivalent across different loss equations.
- [Completed denoising results](../results/denoising_analysis/ANALYSIS_SUMMARY.md): Test 1 and Test 2, with per-battery and early-cycle analyses in their respective reports.

At seed 42, overall errors increased for Test 1 Wang and Test 2 combined, and decreased for Test 1 Zhang. These experiments do not establish a particular window, GCL kernel, FFT operation or mixing coefficient as the cause. Identifying such causes requires separately authorized controlled experiments. This documentation update launches no new training.
