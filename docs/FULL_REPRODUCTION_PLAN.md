# Full MIT reproduction plan

This plan records the training behavior implemented by the authors' unchanged MIT entry point, [`main_MIT.py`](../main_MIT.py), at repository commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`. It is a plan for the next stage; no full training was started while preparing it.

The reproduction will preserve the original PINN architecture, MIT processed dataset, 17 inputs, SOH label, outlier removal, normalization, split rules and three loss equations. New outputs must be stored under `results/reproduction/` so that the authors' bundled results and the completed smoke test remain separate.

## 1. Exact MIT training settings in the code

| Setting | Implemented value | Source and practical meaning |
| --- | --- | --- |
| Maximum epochs | **200 per experiment** | `--epochs=200` in [`main_MIT.get_args`](../main_MIT.py#L8). Early stopping can finish an experiment sooner. |
| Early stopping | **20**, with code stopping when the counter is `> 20` | [`PINN.Train`](../Model/Model.py#L277) validates every epoch, resets the counter on a lower validation MSE and stops after 21 consecutive epochs without a new minimum. |
| Batch size | **512 adjacent-cycle pairs** | `--batch_size=512` in [`main_MIT.get_args`](../main_MIT.py#L8). The last batch may be smaller. |
| Input normalization | **Per-battery min–max scaling to [-1, 1]** | `--normalization_method=min-max`; implemented by [`DF.read_one_csv`](../dataloader/dataloader.py#L43). The SOH target is not min–max scaled. |
| Optimizer | **Two separate Adam optimizers** | [`PINN.__init__`](../Model/Model.py#L127) creates one Adam for `solution_u` and one for `dynamical_F`. Only `lr` is passed, so Adam uses PyTorch defaults for the remaining options: betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`, and AMSGrad disabled. |
| Solution-network initial/warm-up LR | **0.002** | `warmup_lr=2e-3`. The solution Adam is constructed at this learning rate. |
| Solution-network base LR | **0.01** | `lr=1e-2`. This is the top of the warm-up schedule. |
| Solution-network final scheduler LR | **0.0002** | `final_lr=2e-4`. This is the cosine schedule's target value; because of schedule indexing, the final training epoch does not use the exact endpoint. |
| Dynamics-network LR | **0.001, constant** | `lr_F=1e-3`. No scheduler is connected to the dynamics optimizer. |
| Learning-rate scheduler | **30-epoch linear warm-up, then cosine decay for 170 schedule entries** | Custom [`LR_Scheduler`](../Model/Model.py#L94). It is stepped once per epoch after training, with its default `iter_per_epoch=1`. |
| Loss weights | **alpha = 0.5; beta = 0.01** | Total loss is `L_data + 0.5 L_PDE + 0.01 L_physics`. Defaults are in [`main_MIT.get_args`](../main_MIT.py#L29); use is in [`PINN.train_one_epoch`](../Model/Model.py#L237). |
| Validation criterion | **MSE** | Validation MSE is evaluated after every epoch. A strictly lower MSE is considered an improvement. |
| Repeated experiments | **10** | [`main_MIT.main`](../main_MIT.py#L63) loops over `range(10)` and constructs a fresh model and dataloaders for each experiment. |
| Device selection | **CUDA if available; otherwise CPU** | [`Model/Model.py`](../Model/Model.py#L1) selects `cuda` when `torch.cuda.is_available()` is true. The current reproduction environment is CPU-only. |

### Scheduler timing detail

The solution optimizer is initialized at `0.002`. The scheduler is called **after** each epoch, not before it:

```text
train epoch -> scheduler.step() -> validate -> possibly test
```

The warm-up array contains 30 equally spaced values from `0.002` to `0.01`. The cosine array contains 170 values generated from base LR `0.01` toward final LR `0.0002`. Because the update happens after training:

- Epoch 1 trains at `0.002`, then the scheduler sets `0.002` again.
- Epoch 2 also trains at `0.002`, then the scheduler sets the second warm-up value.
- The LR logged beside an epoch is the value assigned after that epoch, which is used by the following epoch.
- The last scheduler value is assigned after epoch 200 and is not used for another training epoch.

An exact reproduction must keep this ordering. Replacing the scheduler with a standard PyTorch scheduler or stepping it at the start of an epoch would change the optimization path.

## 2. Exact train, validation and test construction

The MIT loader reads all three checked-in batches in this order:

1. `2017-05-12`
2. `2017-06-30`
3. `2018-04-12`

For each batch, [`main_MIT.load_MIT_data`](../main_MIT.py#L40) extracts the battery ID from the filename.

### Test batteries

A battery file is assigned to the test set when:

```text
battery ID modulo 5 == 0
```

This gives 23 held-out battery files:

| MIT batch | All battery files | Train/validation battery files | Test battery files |
| --- | ---: | ---: | ---: |
| 2017-05-12 | 46 | 37 | 9 |
| 2017-06-30 | 43 | 35 | 8 |
| 2018-04-12 | 36 | 30 | 6 |
| **Total** | **125** | **102** | **23** |

The test loader contains every adjacent retained-cycle pair from those 23 files. It is not subdivided and is not shuffled.

### Training and validation pairs

For each of the other 102 battery files, the existing loader:

1. Inserts cycle index.
2. Removes rows using the existing three-sigma filter.
3. Divides capacity by the MIT nominal capacity of 1.1 Ah.
4. Normalizes all 17 inputs within that individual battery.
5. Forms adjacent retained-row pairs `(x1, x2, y1, y2)`.
6. Concatenates pairs from all 102 batteries.
7. Applies `train_test_split(..., test_size=0.2, random_state=420)` to the pooled pairs.

The completed smoke-test manifest confirms these effective sizes with the current checked-in data:

| Loader | Adjacent-cycle pairs | Batches at batch size 512 |
| --- | ---: | ---: |
| Training | 54,053 | 106 |
| Validation | 13,514 | 27 |
| Test | 14,170 | 28 |

The training and validation loaders both use `shuffle=True`; the test loader uses `shuffle=False`.

This is a **pair-level** 80/20 train/validation split after pooling the 102 non-test batteries. It is not a battery-level validation split. Pairs from the same battery can occur in both training and validation. The 23 test batteries remain held out at battery-file level.

### File-order dependence

The script uses `os.listdir` without sorting. The split's random index selection is fixed by `random_state=420`, but the battery-pair sequence being indexed depends on the directory listing order. The full run must record the actual ordered file lists in its manifest. Sorting the list would be a split change and must not be introduced during the exact reproduction.

## 3. Random seed status

The authors' MIT training script does **not** set a global random seed for:

- Python's `random` module;
- NumPy;
- PyTorch weight initialization;
- PyTorch `DataLoader` shuffling; or
- deterministic PyTorch algorithms.

The only explicit seed is `random_state=420` inside scikit-learn's train/validation split. Therefore:

```text
train/validation membership: fixed by random_state 420 for a given input order
model initialization:        not seeded
training batch order:         not seeded
dropout masks:                not seeded
```

The ten experiments are consequently repeated stochastic runs rather than ten runs from declared seeds. A later controlled-seed experiment would be useful scientifically, but it would be a separate protocol and must not be presented as the unchanged author-script reproduction.

For auditability, the full run should record that no author seed exists, the ordered battery file lists, and the initial Python/NumPy/PyTorch RNG states or a hash of each state before each experiment. Recording state does not change the random draws. Any later replay that explicitly restores those states must be labeled as a replay mechanism added around the original training code.

## 4. Loss and evaluation behavior to preserve and report

The unchanged training objective is:

```text
L_total = L_data + 0.5*L_PDE + 0.01*L_physics
```

The equations are documented in [`PINN_EXPLAINED.md`](PINN_EXPLAINED.md). They must remain unchanged.

The author loop validates every epoch. Whenever validation MSE reaches a new strict minimum, it immediately evaluates the held-out test set and overwrites `true_label.npy` and `pred_label.npy` for that experiment. The test result does not enter the loss, but repeated test evaluation during training should be disclosed when interpreting the reproduction.

Two implementation details must be reported without silently correcting the baseline:

1. [`eval_metrix`](../utils/util.py#L26) expects `(true_label, pred_label)`, but the author training loop calls it as `eval_metrix(pred_label, true_label)`. MAE, MSE and RMSE are symmetric and are unaffected; MAPE uses the prediction rather than ground truth as its denominator.
2. `self.best_model` is assigned from `state_dict()` when validation improves and saved only after training finishes. PyTorch state dictionaries are shallow mappings to tensors, so the saved `model.pth` may reflect later/final parameter updates rather than the parameters from the validation-best epoch. The prediction arrays are generated at the improvement epoch and can therefore differ from the saved checkpoint's predictions.

The full report should retain the authors' logged values for traceability and separately calculate standard metrics from the saved prediction and target arrays:

- MAE;
- MAPE with ground truth as denominator, reported as both a fraction and percentage;
- RMSE; and
- MSE.

SOH must be identified as a fraction, the sample count must be stated, and metrics should be reported for each experiment plus mean, standard deviation, minimum and maximum across the ten experiments.

## 5. Planned execution procedure

Before launch:

1. Run `git status` and save the source commit plus a patch/hash record of local changes.
2. Run `pip check` and `verify_environment.py` with `.venv-reproduction`.
3. Re-run the bounded smoke test only if the environment, model, loader or training configuration has changed since the successful test on 14 September 2026.
4. Create one timestamped root such as `results/reproduction/MIT_<UTC timestamp>/`.
5. Save the exact arguments, environment versions, CPU/device details, ordered train/validation and test file lists, effective pair counts and RNG-state records before training.

For training:

1. Run ten experiments sequentially with the author settings above.
2. Give each experiment its own directory, `Experiment1` through `Experiment10`.
3. Preserve the existing split construction, unsorted file enumeration, normalization, losses, optimizer and scheduler timing.
4. Capture console output and the existing `logging.txt` for each experiment.
5. Record start/end timestamps, actual epoch count, early-stop epoch, lowest validation MSE and the epoch at which it occurred.
6. Preserve the author-produced checkpoint and arrays, while recording the checkpoint-selection limitation above.
7. Recalculate standard MAE, MAPE, RMSE and MSE from each saved prediction/label pair without replacing the original log.

After training:

1. Confirm every experiment has finite predictions and the expected 14,170 test pairs.
2. Produce a per-run results table and aggregate statistics across all ten experiments.
3. Compare the aggregate with the published MIT result using the same stated metric scale where possible.
4. Label any dependency-related numerical difference: this machine uses the documented Python 3.12/PyTorch 2.14 CPU compatibility environment rather than the authors' Python 3.7/PyTorch 1.7 environment.
5. Keep these baseline outputs separate from all future mixture-of-experts results.

Using a separate runner to manage output directories and manifests is acceptable only if it calls the unchanged loader and `PINN` implementation with these exact settings. Such a runner must not sort files, introduce a seed, change evaluation timing or correct the checkpoint/MAPE behavior inside the baseline. Standard corrected metrics can be added as post-processing and labeled separately.

## 6. Expected runtime on the current CPU environment

The current environment is:

```text
CPU:              AMD Ryzen 7 5800H with Radeon Graphics
Logical CPUs:     16
PyTorch threads:  8 intra-op, 8 inter-op
PyTorch:          2.14.0+cpu
CUDA available:   false
```

The completed three-epoch MIT smoke test used the same 54,053 training pairs, batch size 512, model, derivatives and losses. Its output timestamps span approximately **11 seconds** from model/log creation to final predictions and artifacts. That run performed three training epochs and one test pass, but it did not perform the author's validation pass after every epoch. It is therefore useful as a local scale measurement, not an exact full-run benchmark.

At full settings, each epoch has 106 derivative/backpropagation training batches and 27 validation batches. A test pass of 28 batches also runs whenever validation MSE improves. Data preprocessing is repeated when each of the ten experiments reconstructs its dataloaders.

The practical planning estimate is:

| Scope | Expected CPU wall time | Reasonable planning bound |
| --- | ---: | ---: |
| One experiment reaching all 200 epochs | **about 12–18 minutes** | Includes per-epoch validation, occasional test passes, preprocessing and file output. |
| Ten experiments, all reaching 200 epochs | **about 2–3 hours** | Sequential execution of the author's ten-run loop. Reserve 3 hours to allow for CPU throttling and I/O. |
| Ten experiments with early stopping | **possibly 45–120 minutes** | Actual time depends on when each stochastic run accumulates 21 non-improving epochs. This cannot be known before running. |

These estimates assume the laptop remains plugged in, the CPU is not heavily used by other applications, and thermal/power limits remain stable. OneDrive synchronization may add some output latency. The run should record measured time per experiment so later training can use observed values instead of this estimate.

## 7. Go/no-go checklist before the full launch

- [ ] Original model and dataloader files still match the inspected versions.
- [ ] Compatibility environment passes import and dependency checks.
- [ ] MIT file counts remain 102 train/validation batteries and 23 test batteries.
- [ ] Pair counts remain 54,053 training, 13,514 validation and 14,170 test.
- [ ] Configuration is 200 epochs, batch size 512, alpha 0.5, beta 0.01 and ten experiments.
- [ ] The absence of a global author seed is recorded explicitly.
- [ ] Output root is under `results/reproduction/` and does not already exist.
- [ ] At least three hours of uninterrupted CPU time is available.
- [ ] No model, dataset, feature, label, split or loss changes are included.

No full training was run to create this plan.

## Evidence update from the first full run

On 17 September 2026, inspecting `results/Ours/MIT results.zip` revealed that the bundled historical MIT logs specify `alpha=1` and `beta=50`. The current `main_MIT.py` defaults inspected above specify `alpha=0.5` and `beta=0.01`. Therefore the settings above are verified current-code settings, not proof of the settings that produced the paper's historical results. Run 01 follows the current-code settings exactly as requested. Results must disclose this discrepancy; no loss weight was changed in response to inspecting the archive.

The user authorized only one experiment under `results/reproduction/MIT/run_01/`, with a real best-validation checkpoint. `scripts/reproduce_mit_one.py` calls the original inherited `PINN.Train` and records outputs from the original epoch/validation/test methods. It immediately serializes `best_checkpoint.pth` on validation improvement, while retaining the original deferred `model.pth` behavior. The additional file corrects checkpoint recording only, without changing optimization, selection, test-evaluation timing, or RNG seeding. Standard ground-truth-denominator metrics are reported separately from the original logger's MAPE.
