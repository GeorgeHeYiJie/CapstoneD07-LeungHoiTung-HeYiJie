# Frozen MIT offline denoising comparison

Approved by the user, 28 October 2026. Source commit: `8952464066c66844e624ae5c3d2142296e41b7ba`.

**OFFLINE ONLY:** the supplied denoiser uses centered smoothing and full-trajectory pooling/interpolation. Consequently an early prediction may indirectly use later-cycle features. This is not evidence for real-time deployment or causal online estimation. Capacity is never denoised or used as a denoising input.

## What is compared

Test 1 compares original versus denoised data using `HighDimensionMatrix-PINN_Combination.py` classes/functions. Report Wang and Zhang separately within Test 1. Test 2 compares original versus denoised data using the unchanged `New1HighDimensionMatrix-PINN_Combination.py` model class. A cross-model comparison is not the primary research question.

Only the feature condition differs within each pair. Seed 42, initial weights, battery order, training sample schedule, evaluation samples, capacity/SOH labels, architecture, loss equations, optimizer settings, epoch budget and selection rule are fixed across original/denoised. Scalers are fitted separately on each condition's training features with the same train-only min-max procedure; feature range changes caused by denoising are thus part of the assessed pipeline. The matrix functions receive normalized features in both conditions.

## Frozen data and evaluation

- Exact previous split manifest: 71 train, 18 validation, 36 test batteries; split seed 420. Models use seed 42.
- Same 42,784 training adjacent-pair keys, 13,585 validation keys and 29,083 test keys as the previous analysis. Every evaluated row retains its original CSV row index. Final CSV rows without a subsequent adjacent row remain excluded from evaluation.
- SOH is capacity / 1.1 Ah, in fraction units; labels are taken from the unchanged original CSVs. Denoised CSV capacity values are verified exactly equal after round-trip parsing.
- No 3-sigma filter, label-dependent removal, clipping, row deletion or test-derived normalization. Train-derived feature extrema/medians and maximum training cycle are saved separately. No missing values were present in this dataset.
- All test keys, battery IDs and labels are checked against the existing Wang seed-42 prediction archive. Full test scoring occurs only after 200 epochs and validation checkpoint selection. Smoke tests use train/validation data, not test scores.
- Test 1 Zhang training references use training batteries only, excluding the owner. Validation/test references are frozen across conditions (seeds 20042/30042), two distinct training references per evaluated row, with reference row clamped at that training battery's last available row. This matches the previous frozen evaluation protocol. Reference choices do not depend on feature values or held-out labels.
- MAE, RMSE: fractional SOH; MSE: squared fractional SOH; MAPE: mean absolute error divided by true SOH, multiplied by 100. Each overall metric pools samples. Per-battery metrics are reported separately; batteries are not silently weighted equally in pooled metrics.
- Early-cycle analysis means the first 20% of each original recorded trajectory, used only retrospectively for analysis, never for model inputs.

## Training settings retained and explicit protocol adaptations

All four conditions run for 200 epochs without early stopping or tuning. Each predictor selects its own lowest full-validation prediction MSE checkpoint; Test 1 selects Wang and Zhang independently. Test 2 selects the combined output. These are approved changes from the source entry points' smoke budget/partial validation/linked checkpoint behavior.

Test 1 retains its source default batch size 64, 16-cycle window, two periods, 8x8 matrices, PDE weight 0.5, trend weight 0.01, Zhang mixture 0.5, delta weight 1, dynamics and Zhang Adam learning rates 0.001. Wang uses the source `solution_lr` function: 30-epoch warmup from 0.002 toward 0.01, then cosine toward 0.0002 over a 200-epoch budget. All frozen training adjacent pairs are used (stride 1, the approved full protocol rather than the source smoke stride 5). No authors' model classes are edited. These runs are fresh and are not relabeled versions of the previous batch-512 results.

Test 2 retains its source training granularity: one optimizer update for two consecutive rows, stride 2, sequential frozen training-battery order. This visits non-overlapping pairs from the frozen training trajectories; the exact training-key subset is saved. It retains Adam 0.001, no scheduler, three periods, window 16, 8x8 matrices, PDE weight 0.1, trend weight 0.01 and all network constants. Evaluation is on all frozen validation/test keys, not the source stride-4 partial validation. The unused inter-cell delta head is deliberately left unused, as in the source training loop; enabling it would change the algorithm.

CPU execution uses one intra/inter-op thread consistently for each pair, to avoid thread overhead on New1's very small updates. This is an execution setting, not a tuned model parameter. Runtime includes batch preparation; cached matrix preparation is recorded separately. Hardware and library differences limit comparison with earlier wall times.

## Runtime fixes and isolation

The three source scripts are unchanged and hashed. `scripts/run_denoising_analysis.py` is an isolated adapter:

1. Calls the denoiser's original functions/constants on original feature CSVs, preserving date folders under `prepared/denoised_csv/`. It omits only import-time directory creation and the entry-point call; it does not change numerical functions.
2. Selects original/denoised arrays without copying over source data or rewriting hardcoded source paths.
3. Enables `requires_grad=True` on Test 2 xt inputs before its forward pass, fixing the confirmed physics-derivative runtime error. Loss equations and network parameters are unchanged.
4. Records losses, checkpoints, sample IDs, predictions, configuration and runtime absent from the New1 entry point.
5. Uses per-condition, per-test memory-mapped caches of matrices produced by each source's own matrix function. Caching is a runtime optimization and does not alter representations. The two scripts' different period-selection logic is preserved.

The only expected parameters without gradients in Test 2 are the source's unused `zhang.head_delta` weights/bias. All active gradients and losses must be finite. Each smoke test confirms optimizer updates, finite predictions, checkpoint reload equality and identical initial weights across conditions. Formal models start from fresh seed-42 initialization, not smoke weights.

## Commands

From the repository root, using the existing reproduction environment:

```powershell
.\.venv-reproduction\Scripts\python.exe scripts/run_denoising_analysis.py prepare
.\.venv-reproduction\Scripts\python.exe scripts/run_denoising_analysis.py smoke
.\.venv-reproduction\Scripts\python.exe scripts/run_denoising_analysis.py run --test test1 --condition original
.\.venv-reproduction\Scripts\python.exe scripts/run_denoising_analysis.py run --test test1 --condition denoised
.\.venv-reproduction\Scripts\python.exe scripts/run_denoising_analysis.py run --test test2 --condition original
.\.venv-reproduction\Scripts\python.exe scripts/run_denoising_analysis.py run --test test2 --condition denoised
```

Preparation/smoke/output directories must be fresh; existing outputs are not overwritten. The user explicitly authorized launching all four formal conditions after verification. No previous results, source datasets, main model implementation or historical reproduction outputs are modified.

## Planned result analysis

For each test/predictor: pooled MAE/MAPE/RMSE/MSE, signed difference (denoised minus original), absolute difference magnitude, percentage improvement `(original - denoised)/original*100`, and the same metrics per battery. Positive improvement means lower error. Also report early-cycle metrics, signed bias, extreme-error batteries/samples, share of total/excess error, stability traces, best epoch and runtime.

Use original-vs-denoised matched prediction arrays and retain every test battery. Count improved/worsened batteries and compare pooled with macro-average battery errors so a few outliers cannot be mistaken for broad gains. One seed is descriptive evidence, not a significance claim. Do not infer causal online benefits from this offline transformation.
