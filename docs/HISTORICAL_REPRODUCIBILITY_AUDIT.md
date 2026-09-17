# Historical reproducibility audit

Audited 17 September 2026. No training, optimizer updates, model edits, environment installation or additional reproduction runs were performed for this audit. A sampler-only RNG probe was run in a separate Python process.

## Scope and conclusion

The comparison is between the likely author environment and `results/reproduction/MIT/historical_run_01`, not the current-code baseline. The historical run reconstructs public commit `d72414ac8799747db784203b4c1826ca118dc23a` with archive-evidenced patience 20, alpha 1 and beta 50. The label-only physics loss remains deliberately unchanged: `sum(ReLU(y2-y1))` has no model-parameter gradient.

There is a **confirmed PyTorch sampler behavior difference**, substantial dependency-version differences, and no common recorded training RNG state. Conversely, all 125 current MIT CSVs match the earliest public commit byte-for-byte; archived ordered battery lists and test targets also match. This narrows the investigation toward stochastic optimization and runtime behavior, but does not establish the cause of the accuracy gap.

The archived logs date to July 2023, before the April 2024 initial public source commit. The archives do not include a complete executable environment or initial RNG states. Therefore, matching public source plus logged arguments is not proof of the exact private program used in 2023.

Labels throughout this report mean:

- **confirmed difference**: direct documentary, code or runtime evidence of a difference; its effect on accuracy can still be unknown.
- **likely difference**: strongly expected, but the author-side artifact is missing.
- **possible difference**: a plausible, unverified mechanism.
- **no evidence of difference**: the inspected evidence agrees or shows no relevant change; not a guarantee of bitwise equivalence.

## 1. Environment versions

The author column comes from `README.md`. The current column was verified by importing the installed packages in `.venv-reproduction`, and agrees with the existing environment records. “Confirmed” here refers to the documented stack versus our installed stack; the archived process itself did not log these package versions.

| Item | Documented author environment | Current reproduction environment | Classification and implication |
| --- | --- | --- | --- |
| Python | 3.7.10 | 3.12.14 | **confirmed difference**. Interpreter and supported dependency stack differ; no Python-only numerical defect demonstrated. |
| PyTorch | 1.7.1 | 2.14.0+cpu | **confirmed difference**. Sampling semantics differ directly; CPU kernels, random-number kernels and optimizer implementation may also differ. |
| NumPy | 1.20.3 | 2.5.3 | **confirmed difference**. Used for arrays, normalization-related operations, splitting RNG and the custom LR schedule; exact old-environment outputs have not been compared. |
| pandas | 1.3.5 | 2.3.3 | **confirmed difference**. CSV parsing, statistics and assignment occur at runtime. Current float assignment into the cycle-index column produces a compatibility warning but completed successfully. |
| scikit-learn | 0.24.2 | 1.9.1 | **confirmed difference**. Relevant to `train_test_split`; the inspected split algorithm still uses a seeded permutation. |
| CPU hardware | Supplementary Note 3 describes Intel Core i5-10400F | AMD Ryzen 7 5800H | **confirmed difference** against the described implementation hardware; exact hardware of each archived run is not logged. |
| Numerical libraries and threading | Build, BLAS version and thread count unrecorded | MKL 2026.1, oneDNN 3.12, AVX2-capable build; 8 intra-op and 8 inter-op threads recorded | **likely difference** in build/kernel environment; exact contribution unknown. |
| Execution device | Paper describes CPU implementation; code also permits CUDA | CPU-only torch build | **possible difference** for the archived jobs, whose device is not recorded. There is no evidence that they actually used a GPU. |

The old stack cannot be treated as a dependency downgrade inside Python 3.12: an isolated compatible historical Python environment would be needed. None was installed for this audit. Primary hardware/software reference: [supplementary information, Note 3](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-48779-z/MediaObjects/41467_2024_48779_MOESM1_ESM.pdf).

## 2. Randomness audit

Source: `historical_reproduction/source/main_MIT.py`, `Model/Model.py`, `dataloader/dataloader.py`, and `historical_reproduction/run_mit.py`.

| Source | Observed behavior | Classification |
| --- | --- | --- |
| Python `random` | Imported, but no active MIT training-path random draw or `random.seed` identified. Runner records its initial state. | **no evidence of difference** in an operative Python-random mechanism. |
| NumPy global RNG | No global `np.random.seed` in the author entry point or full historical runner. Split uses a separately seeded legacy RandomState through sklearn. | **no evidence of difference** in seeding policy; no identified uncontrolled NumPy-global draw driving MIT training. |
| `torch.manual_seed` | Neither full-training entry sets it. `random_state=420` seeds the split only, not the networks. Importing the smoke-test helper does not execute its seeded main function. | **no evidence of difference** in policy. |
| Actual initial torch RNG state | Our initial state is saved in `rng_initial.pth`; authors' states are absent. Independent unseeded executions almost certainly start differently. | **likely difference**. Exact matching of initialization and dropout is unavailable. |
| CUDA seeds | No `torch.cuda.manual_seed`/`manual_seed_all` calls or explicit deterministic CUDA configuration. Current run is CPU-only. | **no evidence of difference** in seed policy; CUDA randomness is inactive locally. |
| Model initialization algorithm | `nn.Linear` initialization consumes RNG before explicit Xavier-normal initialization. Solution network reinitializes weights and zeros biases; dynamics MLP reinitializes weights while keeping Linear-created biases. The reconstructed sequence is unchanged. | **no evidence of difference** in source logic. Actual initialized weights are a **likely difference** because training was unseeded. |
| Dropout | Probability 0.2 in solution and dynamics networks. Active during training, disabled by `eval()` in validation/test. Both members of each adjacent-cycle pair are forwarded separately. | **no evidence of difference** in probability/mode logic; realized masks are a **likely difference**. |
| DataLoader shuffle | Train and validation shuffle; test does not. Default generator is `None`. Old and installed RandomSampler implementations consume RNG differently. | **confirmed difference**, detailed below. |
| Worker seeds | `num_workers=0`, no `worker_init_fn`, no worker processes. | **no evidence of difference**; multiprocessing worker seeds cannot explain this run. |
| Iterator base seed | DataLoader iterator construction draws a base seed even with zero workers. Validation and test iterator creation therefore matter to the global stream. Both inspected versions have this mechanism. | **no evidence of difference** in the existence of this draw. Differences in validation improvements can change how often test iterators are created. |
| Repeated-run execution | Historical author entry loops over ten runs without resetting global RNG; our isolated job performs one run. The archives do not prove their launcher was exactly this file. | **possible difference** in process/RNG history; a standalone run is not guaranteed to start at any archived experiment's state. |

### Confirmed sampler change: seed equality would not be enough

In PyTorch 1.7.1, the non-replacement sampler creates a local generator but passes `self.generator` to `randperm`. With the repository's default `None`, the permutation consumes global torch RNG. Installed 2.14 instead passes the newly seeded local generator. Thus batch order and the global state subsequently used by dropout can both change. Source: [PyTorch 1.7.1 RandomSampler](https://raw.githubusercontent.com/pytorch/pytorch/v1.7.1/torch/utils/data/sampler.py), compared with the installed `torch.utils.data.sampler.RandomSampler.__iter__`.

A no-training probe reset torch to seed 420 separately for each path, used 54,053 indices, and emulated only the old sampler's Python branch using current torch primitives:

| Probe output | Old Python branch on current torch | Installed current sampler |
| --- | --- | --- |
| First five indices | 6101, 49484, 42183, 13218, 48899 | 1734, 22669, 47923, 17840, 45340 |
| Post-sampling global RNG SHA-256 | `71213f4823e452a48ad7feeef0d1a81c94c75177700dbb515acc383723f6346d` | `a7da6fee967e2c8c01fd5a6aac4183166e040173d306930f54c7f2779655c8bc` |

This confirms the Python-level distinction without claiming to reproduce old torch kernels or the actual archived permutations. It does not prove either sampling scheme produces worse expected accuracy.

## 3. Data and ordering

| Item | Evidence | Classification |
| --- | --- | --- |
| `os.listdir` | `load_MIT_data` enumerates each batch directory without sorting. Its order is filesystem-dependent. The saved historical ordered train/test lists match all ten archived lists. | **no evidence of difference** in this run; portability risk remains. |
| `glob` | Not used by the active historical MIT data-loading path. | **no evidence of difference**. |
| File sorting | Neither historical loader nor reproduction inserts sorting. Observed lexicographic-looking order is not a source-code guarantee. | **no evidence of difference**. |
| Battery order | Fixed batch order: 2017-05-12, 2017-06-30, 2018-04-12; within batch, enumerated file order. All ten archive path lists agree. | **no evidence of difference**. |
| Within-battery row order | CSV order is retained through cleaning; cycle index is inserted before cleaning. `set(out_index)` deduplicates row-removal indices and does not reorder surviving rows. | **no evidence of difference** in source behavior. |
| Test membership | Battery ID divisible by five is held out. `test_3` uses all adjacent-row pairs in those files; evaluation uses each pair's first sample. | **no evidence of difference**. All ten saved test target arrays are bitwise identical to ours. |
| Train/validation construction | Remaining batteries pooled; adjacent pairs split 80/20 with `random_state=420`, using `train_2`/`valid_2`. Current sizes: 54,053/13,514 pairs; test: 14,170. | **no evidence of difference** in rules. Author-side train/validation tensors were not archived, so equality is not fully proven. |
| sklearn split internals | Old and installed ShuffleSplit use a seeded permutation, take test indices first and train indices next. | **no evidence of difference** in inspected algorithm; old-environment index hashes still need direct verification. |
| Epoch traversal | Training and validation are reshuffled; test is sequential. Validation shuffling affects RNG and potentially floating-point reduction order despite evaluation mode. | **confirmed difference** in sampler implementation, not split membership. |

Relevant source: [scikit-learn 0.24.2 split implementation](https://raw.githubusercontent.com/scikit-learn/scikit-learn/0.24.2/sklearn/model_selection/_split.py). Do not “fix” this investigation by sorting files or changing to battery-level validation: that would change the protocol.

## 4. Adam and learning-rate schedule

| Item | Finding | Classification |
| --- | --- | --- |
| Optimizer and mathematical settings | Two Adam optimizers. Solution starts at 0.002; dynamics stays at 0.001. Default betas (0.9, 0.999), epsilon 1e-8, weight decay 0, AMSGrad false agree between old and current APIs. | **no evidence of difference** in specified algorithm/settings. |
| Gradient clearing | Source calls `zero_grad()` without arguments. Old default `set_to_none=False`; installed default `True`. | **confirmed difference** in API semantics. No demonstrated accuracy effect: connected parameters receive new gradients; missing-gradient parameters would need a diagnostic check. |
| Optimizer implementation | Old Adam uses integer step state; current implementation uses tensor step state and supports additional dispatch paths. Current CPU dispatch check returned fused=false, foreach=false. | **confirmed difference** in implementation, but **possible difference** in resulting numerical updates. Do not attribute this run to fused Adam: that path is not selected here. |
| Floating-point operations | CPU matrix operations, sine activation, autograd through input derivatives and Adam arithmetic span very different torch/build versions. | **possible difference** in accumulated numerical error or optimization stability. No matched-state gradient/update comparison has yet been run. |
| Scheduler implementation | Repository-defined `LR_Scheduler`, not a PyTorch scheduler class. NumPy linspace warmup of 30 entries followed by 170 cosine entries; solution LR only. | **no evidence of difference** in logic/settings. Generic PyTorch scheduler API changes do not apply. |
| Scheduler call timing | `step()` runs after the epoch and before validation. Logged LR is the newly assigned next-epoch rate. Initial optimizer rate and first schedule entry are both 0.002, so epochs 1 and 2 use 0.002. | **no evidence of difference** in reconstructed call order. |
| NumPy schedule rounding | Formula unchanged but generated in newer NumPy. | **possible difference** at numerical precision; not an established major discrepancy. |

Primary old APIs: [Adam 1.7.1](https://raw.githubusercontent.com/pytorch/pytorch/v1.7.1/torch/optim/adam.py) and [Optimizer 1.7.1](https://raw.githubusercontent.com/pytorch/pytorch/v1.7.1/torch/optim/optimizer.py). [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html) also warns that identical seeds do not guarantee identical results across releases/platforms.

## 5. Early stopping and checkpoints

| Item | Finding | Classification |
| --- | --- | --- |
| Patience | Earliest public CLI default is 10, but archived logs specify 20. Our historical wrapper explicitly uses 20. | **no evidence of difference** versus archived configuration; the public-default discrepancy was already corrected explicitly. |
| Selection rule | `PINN.Train`: initialize minimum validation MSE to 10; validate every epoch; update only on strict improvement. Test errors do not choose checkpoints. | **no evidence of difference** in reconstructed logic. |
| Stop condition | Increment counter before training; reset after improvement; stop when counter `> 20`, allowing 21 non-improving epochs. | **no evidence of difference** in logic. |
| Actual outcome | Ours best 79, stop 100, finite training spike 85. Archives mostly reach 200; experiment 2 stops at 193 and experiment 5 at 86. | **confirmed difference** in trajectory/outcome, not evidence of a changed stopping rule. |
| Original deferred checkpoint | `self.best_model` holds shallow state_dict references, serialized after training; tensors can track later updates. Historical runner preserves this original `model.pth` behavior. | **no evidence of difference** in underlying defect. |
| Additional immediate checkpoint | Wrapper additionally saves `best_checkpoint.pth` inside the existing best-epoch Test call. Reload exactly reproduced saved epoch-79 predictions. | **confirmed difference** in recording, not model/training logic. It prevents the deferred-checkpoint defect from corrupting our best-model comparison. |
| Predictions used for comparison | Original code writes prediction arrays immediately on validation improvement. Comparison uses those arrays from both archive and our run, not deferred checkpoint inference. | **no evidence of difference** in selection convention. |

Early stopping can amplify a stochastic/numerical difference: a transient loss spike can prevent further progress before patience expires. It is an observed mechanism, not a proven root cause or a reason to extend patience silently.

## 6. Dataset and preprocessing artifacts

| Item | Evidence | Classification |
| --- | --- | --- |
| Current CSVs versus public paper-era snapshot | All 125 MIT CSVs compared byte-for-byte against Git blobs at `d72414a`; zero mismatches. MIT data history points to the initial public addition. | **no evidence of difference**, strong for the public snapshot. |
| CSVs versus private 2023 run inputs | Raw input hashes were not included in archived logs. Identical ordered lists and test targets support agreement but cannot establish equality of every historical feature. | **possible difference**, with no positive evidence of a mismatch. |
| Precomputed features | Repository provides extracted charging features, not raw curves used to regenerate every feature. Those CSV contents predate our environment and were not recomputed for reproduction. | **no evidence of difference** in supplied artifacts. |
| Feature-extraction provenance | Exact external preprocessing revision, extraction package versions and raw-data-to-CSV manifest are not recorded alongside the archives. Older/private preprocessing is plausible. | **possible difference** in provenance; no evidence our run used newly re-extracted features. |
| Runtime preprocessing source | Historical `DF.read_one_csv`, `delete_3_sigma`, `load_one_battery`, `load_all_battery` retained: inf/NaN removal, per-column three-sigma filtering including capacity, nominal-capacity SOH conversion, per-battery min-max scaling and adjacent pairs. | **no evidence of difference** in source logic, features, target or normalization scope. |
| Runtime preprocessing numerics | Older pandas/NumPy may differ in parsing/reductions/dtype assignment near filtering boundaries. Actual old-stack feature tensors unavailable. | **possible difference**; must compare tensors before blaming training. |
| Locally generated tensor hashes | Historical/current-code runs use identical effective data tensor hashes; all archived test labels match. | **no evidence of difference** locally. This is not proof of old-stack feature-tensor equality. |

Precomputed extraction and runtime normalization are separate stages. Installing old pandas cannot reconstruct unknown raw-feature extraction; it can test whether the unchanged CSVs become the same tensors.

## 7. What the observed discrepancy establishes

All metrics below use SOH fractions, sample weighting over 14,170 test samples and the ground-truth denominator for MAPE. This avoids the separate author-analysis reversed-MAPE/segmentation issues.

| Metric | Historical run 01 | Archived ten-run mean |
| --- | ---: | ---: |
| MAE | 0.0070494441 | 0.0060177859 |
| MAPE (%) | 0.74300128 | 0.63574280 |
| RMSE | 0.0093178301 | 0.0082761733 |
| MSE | 0.0000868220 | 0.0000686079 |

Our four errors exceed the observed archived ranges. One run is insufficient to establish a distributional failure or assign causality to an environment change. The historical label-only loss defect is shared by the reconstruction; its existence is not itself an explanation of the remaining difference.

## 8. Ranked hypotheses and smallest next experiment

The ranking is provisional; hypotheses 1 and 2 overlap through torch RNG.

1. **Uncontrolled initialization and dropout trajectory — likely difference.** No common training seed/state exists, and only one new full run is available. This directly changes optimization and can trigger the observed spike/early stop. There is no proof that sampling variation alone explains the gap.
2. **PyTorch sampler RNG semantics — confirmed difference; accuracy effect unproven.** Even setting the same seed across versions cannot align the default training trajectory. Validation shuffling also participates. The no-training probe has already established this mechanism.
3. **PyTorch/CPU numerical implementation differences — possible difference in effective updates.** The large runtime gap spans Adam internals, dense operations and differentiation through sine networks. Small update differences can grow during training; no matched-tensor experiment has measured them yet.

Changed battery order, changed CSVs, changed loss weights, fused Adam, worker seeding and a mismatched patience setting are not leading explanations given current evidence. Unpublished pre-2024 execution details remain an unresolved provenance limitation.

**Recommend a staged diagnostic, not nine more full runs:**

1. First use a separate historical environment matching the documented five versions, without training. Run the unchanged loader and export ordered paths, retained row IDs, split indices, all four tensors per split, schedule values and initialization hashes. Compare with modern outputs. This resolves preprocessing/split equivalence before optimization is involved. Never overwrite either completed run or silently alter ordering.
2. If tensors agree, run three bounded diagnostic arms for **one epoch each**, retaining the 200-epoch schedule configuration rather than rebuilding a one-epoch scheduler: A = modern runtime/default sampler; B = modern runtime with an isolated adapter reproducing old sampler semantics; C = actual old runtime/default sampler. A and B start from the exact same saved weight tensors and RNG state. Export those same weights to C using a cross-version-readable format; use the same declared seed, without claiming cross-version dropout masks are identical. Keep architecture, losses, batches, split membership and optimizer settings unchanged. The adapter is an explicitly documented diagnostic change to traversal, never a baseline edit.
3. Record first-batch indices, RNG-state hashes, losses, per-network gradient norms and parameter-update differences. For C versus B, first compare a **no-update, eval-mode gradient calculation** on an identical batch and identical weights; eval mode removes dropout only for this numerical diagnostic. It is not an accuracy evaluation or a modified reproduction model. Then examine ordinary train-mode differences. This separates deterministic numerical differences from unresolved random-mask differences as far as possible.
4. To probe sensitivity to initialization, repeat A for one epoch with one second predeclared seed. That makes **four diagnostic epochs total**, with no test-based seed selection. Use train/validation behavior, not test accuracy, for the diagnostic decision.

This is the smallest proposed training screen that includes a sampler control, an actual historical-runtime control, and an initialization sensitivity check. It cannot establish final-accuracy distributions or explain an epoch-85 event conclusively. If differences appear, identify the earliest divergent operation before approving any longer paired comparison. If the old environment is unavailable, A/B plus the second seed can still be informative, but cannot validate historical kernel behavior.

No part of this recommended training experiment was executed during this audit. Keep historical run 01 as evidence, keep current-code Run 01 untouched, and leave runs 02–10 unstarted.

## Evidence map

- `README.md`, `ENVIRONMENT.md`, `requirements-reproduction.txt`: documented and installed stack.
- `docs/LOSS_WEIGHT_DISCREPANCY.md`, `docs/HISTORICAL_REPRODUCTION.md`: commit reconstruction and archive attribution limits.
- `historical_reproduction/source/`: inspected historical model, loader, entry point and utility code.
- `historical_reproduction/source_manifest.json`: historical source provenance.
- `results/reproduction/MIT/historical_run_01/run_configuration.json`, `rng_initial.pth`, `dataset_sha256.json`: local configuration and state records.
- `results/reproduction/MIT/historical_run_01/archive_comparison.json`: all ten archived argument sets, ordered path lists, target equality and metric comparison.
- `results/reproduction/MIT/historical_run_01/loss_history.csv`, `metrics.json`, `test_history.json`: optimization trajectory, checkpoint verification and result selection.
- `results/Ours/MIT results.zip`: archived author artifacts.

Audit changes are limited to this document. Existing model implementations, datasets and completed results were not modified.
