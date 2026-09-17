# FYP repository instructions

These instructions apply to all work in this repository.

## Project goal

First reproduce the published PINN4SOH results for battery state-of-health (SOH) estimation. Then develop and evaluate a simplified physics-informed mixture-of-experts model for battery degradation prediction. Keep the original PINN as the reference baseline throughout the project.

Read `REPRODUCTION_NOTES.md` before implementation or experiment work. It documents the inspected source revision, data pipeline, losses, execution traps and known evaluation issues. Check the current code before assuming a documented issue is still present.

## Preserve the authors' implementation

- Preserve the original authors' model code wherever possible. Prefer separate wrappers, configuration files and new modules over rewriting the baseline.
- Do not replace or redesign the PINN architecture during reproduction. Put the future mixture-of-experts implementation in a separate module and give it a separate entry point.
- Keep original datasets, supplied checkpoints and published-result artifacts intact. Do not overwrite them with new experiments.
- Distinguish an exact attempt to reproduce the authors' implementation from a corrected evaluation or a new modeling experiment. Label deviations explicitly.

## Data and experiment integrity

- Never silently change training, validation or test splits. Record battery IDs, the split procedure, random seeds and any dependence on file ordering.
- Never silently change features or labels. Record feature names and order, cleaning rules, normalization, capacity divisors and target definitions.
- Treat changes to preprocessing, pair construction, normalization statistics or evaluation sample selection as experiment changes, even if described as bug fixes.
- Explain proposed changes to these choices before implementing them, and document the original and revised behavior. Follow explicit user constraints; do not introduce an extra permission step for routine work already authorized.
- For comparisons, use the same data protocol and metric definitions where possible. If protocols differ, make the difference visible and do not present the results as a controlled model comparison.

## Compatibility changes

- Keep Python, PyTorch and dependency compatibility modifications minimal and preserve the intended numerical behavior wherever possible.
- Document every compatibility modification in `REPRODUCTION_NOTES.md`: affected files, reason, old and new behavior, environment versions, validation performed and any expected effect on results.
- Do not present a scientific or evaluation change as a compatibility-only fix. Record any numerical or protocol deviation separately.

## Training and saved results

- Run and pass a small smoke test before any long training run. Repeat it after changes that affect the model, data pipeline, training configuration or environment.
- Bound the smoke test explicitly by epochs, batches or experiment count. Inspect entry-point loops so a short epoch setting does not accidentally launch many experiments.
- Check input and target shapes, finite predictions and losses, a successful optimizer update, validation/test execution and expected saved outputs. A smoke test establishes execution, not reproduction accuracy.
- Report failures and resolve relevant issues before launching long training. Honor any user instruction not to start training.
- Save new PINN reproduction outputs under `results/reproduction/`, smoke tests under `results/smoke_tests/`, and future mixture-of-experts outputs under `results/pimoe/`.
- Give each run a distinct directory. Do not overwrite earlier runs or the authors' bundled artifacts. If a run uses a different output location, record it explicitly.
- Save the source commit and local changes, configuration, environment versions, seeds, battery split manifest, logs, predictions, labels and checkpoint information needed to trace the run.
- Record whether predictions and checkpoints correspond to the final epoch or the best validation epoch. Verify checkpoint-saving behavior before using a checkpoint as the reference baseline or for transfer learning.

## Evaluation

- Report **MAE, MAPE, RMSE and MSE** for every completed evaluation, with the dataset, split, sample count and aggregation method. Identify smoke-test metrics as preliminary.
- For standard metrics, use ground-truth labels as `y_true` and predictions as `y_pred`. Calculate MAPE relative to ground truth and state whether it is reported as a fraction or a percentage. If a zero target occurs, document how it is handled.
- State whether SOH is represented as a fraction or percentage. MAE and RMSE use that scale; MSE uses its square.
- The inspected authors' code has a reversed-argument MAPE issue. Do not silently replace historical metric values: distinguish metrics produced by the original implementation from corrected standard metrics and document the correction.
- Select checkpoints and tune settings using training/validation data. Do not use test performance to choose a model or configuration.
- Keep per-run results. When reproducing repeated experiments, report their aggregate and variability, and distinguish sample-weighted metrics from averages across batteries.

## Communication

- Explain major code changes in simple engineering language: what changed, why it was needed, how it affects battery prediction and how it was checked.
- Separate verified results from assumptions, expected behavior and untested proposals.
- When reporting work, identify changed files, validation performed and remaining limitations. Do not claim that a successful smoke test reproduces the paper's published accuracy.
