# Denoising comparison preflight

Inspected 28 October 2026. Shared repository: GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie. Local main and freshly fetched group/main both point to `8952464066c66844e624ae5c3d2142296e41b7ba`.

The existing untracked `tmp/` PDF-export files were preserved. No model source, parameters, original data, or previous experiment results were changed. No optimizer updates or training runs were performed.

## Confirmed comparison scope

- Test 1: HighDimensionMatrix-PINN_Combination.py, original versus denoised features. Report its Wang and Zhang outputs separately within this test.
- Test 2: New1HighDimensionMatrix-PINN_Combination.py, original versus denoised features.
- The main comparison is within each test, not between the two scripts.
- Algorithm-structure presentation material belongs to the teammate; this work covers execution and result analysis only.

## File and runtime checks

All three requested scripts exist in the repository root. Python 3.12 and the installed NumPy, pandas, SciPy, PyTorch, scikit-learn and matplotlib imported successfully in the preceding inspection. The 125 MIT CSVs have the required 16 feature columns and capacity; their numeric data were finite in that inspection.

The existing combination script has a guarded entry point and an existing runnable baseline. It was not retrained during this check.

New1 successfully produced two predictions with matrix shape [2,3,16,8,8], but its physics residual raised `One of the differentiated Tensors does not require grad`. Its training loop creates xt without requires_grad. A diagnostic input with requires_grad=True produced a finite residual of shape [2,1]. This diagnostic did not change source files or perform an optimizer update. A documented runtime correction is needed before running Test 2.

DenoisingAlgorithm.py currently points to the nonexistent `./your_raw_csv_folder`. It processes CSVs in one directory, not recursively, and writes the same filenames/columns to OUTPUT_DIR. Capacity is retained. Importing it also creates its default output directory, so it was not imported during this preflight.

## Data flow to implement before execution

Keep `data/MIT data/` untouched. Prepare a separate denoised tree with the same three date folders and battery filenames. Call the existing denoising functions once per source date folder with separate DATA_DIR/OUTPUT_DIR values; preserve all denoising constants. Verify filenames, row counts, capacity values and finite outputs against the originals.

Test 1 reads a hardcoded ROOT/data/MIT data tree and has no --data-dir flag. It needs a path-only adapter or isolated script workspace to select original versus denoised trees without overwriting source CSVs. Merely changing the working directory does not work because ROOT comes from the script location.

Test 2 accepts --data-dir and can select either CSV tree directly. Both scripts build matrices internally. Neither expects a precomputed matrix file; DenoisingAlgorithm.py outputs feature CSVs, not matrices.

Use separate original/denoised run directories under test1/ and test2/, including separate working directories for New1's fixed best_combined.pt output. Preserve each model's hyperparameters within its paired comparison.

## Decisions required for valid result commands

1. Test 1 defaults to 2 epochs, capped batteries (3/1/1), batch size 64 and stride 5. Those are smoke settings, not the previous full frozen evaluation. Test 2 defaults to 50 epochs. Do not silently replace these settings with earlier experiments' settings; agree the formal budget explicitly.
2. Test 2 has an 80/20 train/validation split and no independent test set. Its checkpoint score is summed absolute error over the first five validation batteries at stride 4. Choose either an explicit frozen train/validation/test evaluation adapter or honestly label its existing selected-checkpoint results as validation-only.
3. Test 1 filters 3-sigma outliers using all columns including capacity. Denoising may change retained rows and cycle numbering. Comparisons must audit original row identities and disclose different evaluation samples; a fixed-row protocol would be an explicit preprocessing change.
4. Both scripts normalize with full-battery extrema, not the earlier train-only frozen normalization. The denoiser also uses centered smoothing and full-trajectory pooling/interpolation. Even a causal matrix window can therefore contain future-cycle information after denoising. Results under unchanged preprocessing must be described as retrospective/offline, not causal online prediction.
5. Both models use seed 420 by default, but New1's filesystem order is unsorted. Freeze and save a common battery order/split for each original/denoised pair. Retain distinct output paths so checkpoints cannot overwrite one another.

No runnable *formal comparison* command is certified yet: the path adapter, New1 runtime correction, evaluation decision and epoch budget must be settled first. Running the raw entry points now would either use smoke defaults or fail, and would not fulfil the requested result collection.

## Required outputs after runs

For each test, save configuration, data/split manifests, seed, checkpoint/best epoch, loss/validation history, runtime, predictions with battery and original-row IDs, and MAE/MAPE/RMSE/MSE. Test 1 must retain separate Wang and Zhang metric rows.

Report signed difference `denoised error - original error` and its absolute magnitude; report improvement as `(original error - denoised error) / original error * 100%` (undefined if original error is zero). Positive improvement means lower denoised error.

Produce overall and per-battery tables, prediction plots, paired per-battery MAE changes, early-cycle comparisons, outlier and stability observations, and a short results-only Markdown conclusion. Distinguish broad gains from gains dominated by a small number of outliers. Preserve all previous high-dimensional comparison results.
