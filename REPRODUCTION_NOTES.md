# PINN4SOH reproduction notes

Inspected on 14 September 2026. Repository: `smolcatshark/PINN4SOH`, commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`.

This document describes the initial repository inspection; subsequent environment changes are recorded in the compatibility log at the end. During the initial inspection no model, training script, or dataset was modified, and no training was started. The review covered all Python source files and repository documentation, the directory inventory, and headers of all 387 CSV files. Saved ZIPs, spreadsheets, images and model weights were inventoried as supporting artifacts; their numerical results and checkpoint compatibility were not independently validated.

The main flow is:

```text
Pre-extracted charging features in one CSV per battery
  -> cleaning, SOH conversion, feature scaling
  -> adjacent-row pairs and dataset splits
  -> solution_u: 17 inputs -> one SOH prediction
  -> train with data error + dynamics residual + trend penalty
```

## 1. Where the PINN architecture lives

`Model/Model.py` is the central file:

| Class/method | Responsibility |
| --- | --- |
| `Sin` | Sine activation, `torch.sin(x)` |
| `MLP` | Reusable fully connected network used by the encoder and dynamics model |
| `Predictor` | Final SOH prediction head |
| `Solution_u` | Maps the 17 input values to one scalar SOH |
| `PINN.__init__` | Combines `solution_u` and `dynamical_F`, creates optimizers and loss functions |
| `PINN.forward` | Computes the SOH prediction, automatic derivatives and dynamics residual |
| `PINN.train_one_epoch` / `Train` | Loss calculation, optimization, validation, stopping and saving |
| `PINN.predict`, `Valid`, `Test` | Inference through `solution_u` only |
| `LR_Scheduler` | Linear warmup followed by cosine learning-rate decay |

The actual solution network has linear-layer widths **17 -> 60 -> 60 -> 32 -> 32 -> 1**. The first three linear layers form the encoder; the last two form the predictor. Sine activations follow the two 60-unit layers and the predictor's 32-unit layer. Dropout with probability 0.2 occurs after the second encoder activation and before the predictor's first linear layer. There is no sigmoid or clipping on the output.

The dynamics network defaults to **35 -> 60 -> 60 -> 1**, with sine activations and dropout. Its depth and width come from `F_layers_num` and `F_hidden_dim`. Its 35 inputs are the concatenation `[xt, u, u_x, u_t]`: 17 original inputs, one predicted SOH, 16 feature derivatives and one cycle derivative.

`main_adaptation - fine-tuning.py` defines `AdaPINN`, a subclass that loads a checkpoint, freezes the dynamics network and updates the solution network. It is for transfer experiments, not the initial reproduction.

`Model/Compare_Models.py` contains the MLP and CNN baselines. These are not the PINN itself. The baseline MLP reuses the encoder and predictor definitions but trains with data loss alone in `main_comparision.py` (the filename really is spelled “comparision”).

## 2. Dataset loading and preprocessing

`dataloader/dataloader.py` implements the common `DF` loader and dataset-specific `XJTUdata`, `HUSTdata`, `MITdata` and `TJUdata` classes. The main scripts choose which battery files to pass to these classes.

The checked-in data are already feature tables:

| Folder | CSV/battery count |
| --- | ---: |
| `data/XJTU data` | 55 |
| `data/HUST data` | 77 |
| `data/MIT data` | 125 |
| `data/TJU data` | 130 |
| Total | 387 |

All CSV headers have the same ordered 16 feature columns followed by `capacity`. Raw voltage/current time-series extraction is **not implemented in this repository**. The README links the authors' separate [preprocessing code library](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library); that external library was not audited here.

`DF.read_one_csv` performs these steps, independently for each battery:

1. Read the CSV and insert `cycle index` immediately before `capacity`. The index is `0, 1, ..., N-1`, generated from row order before cleaning, not read from an original experimental cycle-ID column.
2. Replace positive/negative infinity with NaN and drop rows containing NaN.
3. Remove any row lying outside mean +/- three standard deviations in any column. This also includes the capacity target. Statistics are computed on the battery's available rows before the outlier deletion.
4. Divide capacity by the dataset's nominal capacity to obtain SOH.
5. Scale all 17 input columns, including cycle index. The main scripts default to min-max scaling to [-1, 1]: `2*(value-min)/(max-min)-1`. The alternative is z-score scaling: `(value-mean)/std`, using pandas' standard deviation.

Scaling happens when `nominal_capacity` is supplied, as it is in the normal dataset classes. The target is not min-max/z-score standardized. There is no explicit protection against zero feature range or zero standard deviation after cleaning.

`DF.load_one_battery` creates `(x1, y1)` from all retained rows except the last, and `(x2, y2)` from all retained rows except the first. Thus the model learns from pairs of successive retained rows of the same battery. After outlier removal these need not be successive original cycles. A battery with N retained rows contributes N-1 pairs; pairs never cross battery boundaries.

`DF.load_all_battery` concatenates pairs across batteries, converts them to float32 tensors, and packages `(x1, x2, y1, y2)` into PyTorch `TensorDataset` / `DataLoader` objects. Feature tensors have shape `[number_of_pairs, 17]`; labels have shape `[number_of_pairs, 1]`.

## 3. What the 17 inputs represent

The following order is verified from the CSV headers and loader insertion order. It is the model's actual input order, not a reordered conceptual feature list.

| Input number | Column | Meaning |
| ---: | --- | --- |
| 1 | `voltage mean` | Average voltage in the selected charging segment |
| 2 | `voltage std` | Spread of voltage values |
| 3 | `voltage kurtosis` | A statistic describing the shape/tails of the voltage distribution |
| 4 | `voltage skewness` | Asymmetry of the voltage distribution |
| 5 | `CC Q` | Accumulated charge in the selected constant-current charging segment |
| 6 | `CC charge time` | Duration of that segment |
| 7 | `voltage slope` | Slope of the selected voltage curve |
| 8 | `voltage entropy` | Entropy statistic of that curve; not thermodynamic entropy |
| 9 | `current mean` | Average current in the selected charging segment |
| 10 | `current std` | Spread of current values |
| 11 | `current kurtosis` | Shape/tails of the current distribution |
| 12 | `current skewness` | Asymmetry of the current distribution |
| 13 | `CV Q` | Accumulated charge in the selected constant-voltage charging segment |
| 14 | `CV charge time` | Duration of that segment |
| 15 | `current slope` | Slope of the selected current curve |
| 16 | `current entropy` | Entropy statistic of that curve |
| 17 | `cycle index` | Generated row-based cycle coordinate, then normalized |

For the physical context, the paper selects voltage data in the last 0.2 V below the charge cutoff and current data between 0.5 A and 0.1 A during constant-voltage charging. These are charging-curve features, not post-charge relaxation features. See the paper's [Feature extraction section](https://www.nature.com/articles/s41467-024-48779-z).

The CSVs do not define the exact entropy estimator, slope fitting procedure, kurtosis convention, or original time/charge units for every dataset. Those details require the supplementary material and external raw-data preprocessing code. This loader consumes the stored numbers; it does not recompute those statistics. In particular, `CC Q` and `CV Q` are feature values and are distinct from the final `capacity` target.

In `PINN.forward`, `x = xt[:, :-1]` means the first 16 features and `t = xt[:, -1:]` means the cycle coordinate. Here t is not elapsed time in seconds, and temperature is not a separate input.

## 4. Prediction target

The supervised target is one scalar **state of health (SOH)** at the supplied cycle:

```text
y = CSV capacity / nominal capacity
```

| Dataset | Capacity divisor used by the code |
| --- | ---: |
| XJTU | 2.0 Ah |
| HUST | 1.1 Ah |
| MIT | 1.1 Ah |
| TJU Dataset 1 / Dataset 2 / Dataset 3 | 3.5 / 3.5 / 2.5 Ah |

For example, 1.8 Ah / 2.0 Ah = 0.90, meaning 90% SOH. The code uses a fixed nominal capacity, not each battery's first measured capacity. SOH can exceed 1, and the prediction head does not constrain its range.

The supplied training task estimates SOH from that cycle's features and index. Pairing rows does not turn it into a next-cycle forecasting model: `u1` is compared with `y1` and `u2` with `y2`. It does not directly output remaining useful life or total cycle life, nor provide an autonomous future rollout without future feature inputs.

## 5. Training, validation and test construction

### Main experiment route

The ordinary dataset scripts first hold out battery files for testing. On the remaining batteries they select `train_2` and `valid_2` from the loader: an **80%/20% random split of pairs**, using `train_test_split(..., test_size=0.2, random_state=420)`. Test files are loaded separately and their complete pair collection is selected through `test_3`.

| Script | Test-battery selection |
| --- | --- |
| `main_XJTU.py`, `load_data` | Within the selected batch, any filename containing `4` or `8`. For 2C this holds out batteries 4 and 8; six batteries supply training/validation. For 3C it also holds out battery 14, giving 12 training/validation and 3 test batteries. Other batches have 6 and 2. |
| `main_HUST.py`, `load_HUST_data` | Explicit IDs: 1-4, 1-8, 2-4, 2-8, 3-4, 3-8, 4-4, 4-8, 5-4, 5-7, 6-4, 6-8, 7-4, 7-8, 8-4, 8-8, 9-4, 9-8, 10-4, 10-8. There are 57 remaining batteries and 20 test batteries. |
| `main_MIT.py`, `load_MIT_data` | Across all three date batches, parsed battery ID divisible by 5. The current files give 102 remaining batteries and 23 test batteries. |
| `main_TJU.py`, `load_TJU_data` | In the same-batch route, enumerate `os.listdir` filenames with a 1-based position. Hold out positions whose last digit is (5,9), (4,8), or (5,9) for batch indices 0,1,2 respectively. This uses directory position, not the numeric battery filename. |

TJU also selects its batch directory through unsorted `os.listdir`. Both directory ordering and file ordering can therefore change the selected identities. The class's named datasets contain 66, 55 and 9 files respectively. The cross-batch route is separate and has indexing problems described below.

Training and validation loaders shuffle their pairs. Test loaders do not. `Valid` and `Test` only use `x1` and `y1`, so the last retained row of each test battery is absent from its evaluated predictions.

### Other loader options

The same loader also returns `train`, `valid`, and `test`: it takes the first 80% of the **concatenated pair collection**, then randomly splits that portion 80/20 for training/validation, leaving the last 20% as test. The approximate fractions are 64/16/20. This is not necessarily a per-battery chronological split. The ordinary main scripts use `train_2`, `valid_2`, and `test_3` instead, so these alternative keys do not describe their experiment protocol.

### Interpretation and reproducibility limits

- Validation shares battery identities with training. Adjacent pairs overlap, so a row can occur in a training pair and a validation pair. This is not validation on unseen batteries.
- Cleaning and normalization use the full trajectory of each battery before pair splitting, including the held-out test batteries' own trajectories. Outlier filtering also consults capacity labels. This reproduces the supplied offline protocol but should not be described as strictly causal online prognosis or preprocessing fitted on training data only.
- A fixed split seed does not fix weight initialization, dropout or minibatch shuffling. The main scripts do not set global random seeds. Unsorted file lists also affect concatenation and small-sample selections.
- Validation MSE is checked each epoch. When it improves, the test set is evaluated and predictions are saved. The improvement decision uses validation, not test error. Early stopping occurs when the counter is **greater than** `early_stop` (default 20), i.e. after 21 consecutive non-improving epochs under the normal route.

## 6. The three losses, exactly as implemented

Source: `PINN.forward` and `PINN.train_one_epoch` in `Model/Model.py`. Let a minibatch contain B paired samples, with predictions u1/u2 and measured SOH labels y1/y2.

**Data loss: fit the measured SOH.**

```text
L_data = 0.5 * mean((u1-y1)^2) + 0.5 * mean((u2-y2)^2)
```

This is `nn.MSELoss()` applied to both members of each pair.

**PDE/dynamics loss: match a learned derivative model.**

Automatic differentiation computes `u_t = partial u / partial t` and the 16 components of `u_x = partial u / partial x`. `create_graph=True` keeps these derivative calculations differentiable during optimization.

```text
F = dynamical_F(concat(xt, u, u_x, u_t))
f = u_t - F
L_PDE = 0.5 * mean(f1^2) + 0.5 * mean(f2^2)
```

Thus the residual is pushed toward zero. The derivative is with respect to the normalized cycle coordinate. This implementation learns the right-hand side with a neural network; it does not insert a full electrochemical model. Notice that F itself receives u_t as an input, so this residual is not an independent first-principles check on the learned derivative.

**Physics/trend loss: discourage a predicted change opposite to the measured change.**

```text
L_physics = sum(ReLU((u2-u1) * (y1-y2)))
```

If measured SOH decreases, y1-y2 is positive, so a predicted increase is penalized. If measured capacity recovers, the sign reverses and a predicted decrease is penalized. Equal labels yield no penalty. For example, labels 0.95 -> 0.94 and predictions 0.93 -> 0.94 contribute `0.01 * 0.01 = 0.0001` to the sum.

This is a label-dependent, soft direction penalty, not a hard requirement that SOH always decrease. It is a **sum**, whereas the other losses use **means**, so its relative influence depends on batch size and label differences.

```text
L_total = L_data + alpha * L_PDE + beta * L_physics
```

| Main script | alpha | beta | Batch size | Dynamics learning rate |
| --- | ---: | ---: | ---: | ---: |
| XJTU | 0.7 | 0.2 | 256 | 0.001 |
| HUST | 0.6 | 0.1 | 512 | 0.0005 |
| MIT | 0.5 | 0.01 | 512 | 0.001 |
| TJU | 0.1 | 0.1 | 512 | 0.001 |

Two Adam optimizers update `solution_u` and `dynamical_F` after a shared backward pass. The solution learning rate uses the scheduler; the dynamics rate stays fixed. Default main-script settings are 200 epochs, 30 warmup epochs, warmup LR 0.002, base LR 0.01, final LR 0.0002. Scheduler steps occur at the end of each epoch. Logged epoch loss components are averages over batches without weighting the final smaller batch by its size.

The local `fix.md`, dated September 2026, says bug fixes caused alpha and beta to differ from the Supplementary File. Record this commit and its actual arguments when reporting results; do not silently replace them with historical paper values.

## 7. Simplest first reproduction

Use **the XJTU 2C case through `main_XJTU.py`'s existing functions**. It is the README's demo dataset, has only eight batteries in that batch, and is a straightforward way to understand the pipeline.

However, executing `python main_XJTU.py` directly runs all six batches with ten repetitions each: **60 training runs**, each allowing up to 200 epochs. Its `main()` overwrites `--batch` and `--save_folder`. Passing `--batch 2C` does not limit that loop.

For a future one-experiment, two-epoch smoke test, the following PowerShell block imports the existing functions without calling the experiment loop. It changes run configuration only; it leaves every architecture definition intact. **This command was documented, not executed.** Run it from the `PINN4SOH` repository directory in the configured Python environment:

```powershell
@'
import os
import sys
from main_XJTU import get_args, load_data
from Model.Model import PINN

sys.argv = ["first_reproduction"]
args = get_args()
args.batch = "2C"
args.epochs = 2
args.warmup_epochs = 1
args.save_folder = "results/first_reproduction_XJTU_2C"
args.log_dir = "logging.txt"
os.makedirs(args.save_folder, exist_ok=True)

loaders = load_data(args)
print({name: len(loader.dataset) for name, loader in loaders.items()})
pinn = PINN(args)
pinn.Train(trainloader=loaders["train"],
           validloader=loaders["valid"],
           testloader=loaders["test"])
'@ | python -
```

The output directory must exist before loading data because the loader writes its file list there. The shortened warmup matches the short run. A smoke test only verifies execution and output generation; it is not evidence of reproducing the paper's accuracy. A subsequent single full experiment would restore 200 epochs and 30 warmup epochs and use a new output directory. Reproducing reported averages requires the corresponding repeated experiments and verified evaluation protocol.

Expected outputs are a log, `true_label.npy`, `pred_label.npy`, and `model.pth`. These are only expected outputs; none were generated by this inspection.

Other entry points are less suitable initially: HUST and MIT `main()` each run ten repetitions, while `main_TJU.py` and the adaptation script currently end in `pass`, so running them directly does not start their defined training routines. The parameter-count script only instantiates networks and reports counts.

## 8. Dependencies and versions

The authors' README explicitly specifies this historical environment:

| Component | Documented version | Purpose |
| --- | --- | --- |
| Python | 3.7.10 | Interpreter |
| PyTorch (`torch`; conda package `pytorch`) | 1.7.1 | Networks, automatic differentiation, training |
| scikit-learn (import name `sklearn`) | 0.24.2 | Splits and regression metrics |
| NumPy | 1.20.3 | Arrays, learning-rate schedule, saved predictions |
| pandas | 1.3.5 | CSV loading and tabular results |
| Matplotlib | 3.3.4 | Analysis and plotting |
| SciencePlots (`scienceplots`) | Not pinned | Plot styles; not needed for the core training path |

These are the authors' documented versions, not a proven minimum-version specification or a verified modern environment. There is no `requirements.txt`, conda environment file, lockfile, or automated test suite in the checkout. Cached `.pyc` filenames referring to newer Python versions do not establish supported versions.

The minimal training path imports torch, numpy, pandas and scikit-learn, plus Python standard-library modules. Other utilities additionally require:

- `seaborn` for Figures 4b and 5.
- `openpyxl`, explicitly selected by the Excel-reading plotting scripts; Excel export also needs a suitable pandas Excel writer engine.
- `pyautogui` and `keyboard`, imported by the optional `plotter/color.py` utility only.

Those extra packages are not version-pinned in the README. Some analysis scripts use older `ExcelWriter.save()` APIs, so swapping in a current pandas release should not be assumed to work unchanged. Plotting styles may also have external rendering requirements; plotting was not tested.

The model selects CUDA when `torch.cuda.is_available()` is true, otherwise CPU. A GPU is not mandatory. No CUDA toolkit/driver version is pinned in the README. XJTU/MIT/TJU scripts assign `CUDA_VISIBLE_DEVICES='0'`; HUST assigns `'1'`, after importing the model module, so device selection deserves checking on the eventual machine.

No environment was installed or runtime compatibility tested in this task. On the current shell PATH, `python` resolves to the WindowsApps alias and `conda` was not found; this is not proof that Python/conda are absent elsewhere on the computer.

## Additional findings relevant to later reproduction

These observations were documented without applying fixes:

- **Best-checkpoint storage:** `Train` stores plain `state_dict()` objects in `self.best_model`, without deep copying their tensors, then saves after training. Those tensors can continue changing with training. The final `model.pth` may therefore not represent the epoch that produced the saved best-validation predictions. Verify this before using checkpoints for inference or transfer.
- **MAPE argument order:** `utils/util.py` defines `eval_metrix(true_label, pred_label)`, but PINN training and several analysis scripts call it as `eval_metrix(pred_label, true_label)`. MSE, RMSE and MAE are symmetric; MAPE is not. The logged MAPE uses predictions in the denominator. Account for this when comparing metrics with a conventional ground-truth-denominator MAPE.
- **TJU cross-batch indexing:** `TJUdata.read_one_batch` accepts 1,2,3 and indexes folder names by `batch-1`, but indexes capacities by `batch`. This selects wrong capacities for some batches and is out of range at 3. `main_TJU.py` also exposes 0-based batch arguments. Its normal same-batch `read_all` route chooses capacity by folder name and avoids this particular method.
- **Output-path mismatch:** current XJTU training writes under `results of reviewer/XJTU results`, and MIT under `202608/Push/MIT results`. Some analysis scripts and README instructions expect other folders. Check paths before running analysis.
- **Analysis assumptions:** result parsers infer battery boundaries from jumps in SOH instead of saved battery IDs per prediction. Their slices can omit boundary samples. `FineTune results.py` additionally attempts to unpack five metrics, while `eval_metrix` returns four. Do not treat all analysis scripts as ready to run unchanged.
- **README filename typo:** the README calls the baseline script `main_comparison.py`; the actual file is `main_comparision.py`.
- **Current versus historical artifacts:** bundled result archives and pretrained checkpoints may come from earlier runs. Their presence does not verify that the current revised source reproduces those exact values.

## Remaining repository map

| Files/folders | Role |
| --- | --- |
| `utils/util.py` | Logging, averaging losses, evaluation metrics, appending data paths to logs |
| `data analysis/sample count.py` | Counts CSV rows and batteries; does not extract raw features |
| `count parameters.py` | Instantiates PINN and baselines and prints parameter counts; its PINN count covers `solution_u` only |
| `results analysis/{XJTU,HUST,MIT,TJU} results.py` | Parse dataset experiment logs/predictions and export summary spreadsheets |
| `results analysis/Comparision results.py` | Baseline result summaries |
| `results analysis/FineTune results.py` | Transfer-experiment summaries |
| `plotter/Figure 2.py` | Battery capacity trajectories |
| `plotter/Figure 4a.py` | Predicted versus measured SOH |
| `plotter/Figure 4b.py` | Regular-experiment metric comparisons |
| `plotter/Figure 5.py` | Small-sample metric comparisons |
| `plotter/color.py` | Optional color utility |
| `pretrained model` | Saved model checkpoints and auxiliary NPZ arrays; ordinary training does not load these |
| `results` | Bundled PINN/MLP/CNN result archives |
| `results analysis/processed results*` | Existing summary spreadsheets |
| Root PNG images | Dataset illustrations used by README |
| `__init__.py`, `__pycache__` | Package markers and generated bytecode, not separate model implementations |

At the initial inspection, environment setup and a bounded smoke test were the next steps. Environment setup has since been completed as recorded below; a training smoke test has not yet been performed.

## Compatibility log: environment setup, 14 September 2026

Created `requirements-reproduction.txt`, `ENVIRONMENT.md`, `verify_environment.py` and `.gitignore`. The isolated `.venv-reproduction` uses Python 3.12.14 and CPU PyTorch 2.14.0+cpu. The original pinned stack has no matching Python 3.12 Windows wheels; the torch 1.7.1 pip dry-run failed. The dependency-by-dependency rationale and recreation commands are in `ENVIRONMENT.md`; all 45 installed distributions are pinned in the requirements file. Original transitive versions were not recorded by the authors and are not invented here.

Direct dependency changes: torch 1.7.1 -> 2.14.0+cpu; NumPy 1.20.3 -> 2.5.3; scikit-learn 0.24.2 -> 1.9.1; pandas 1.3.5 -> 2.3.3; Matplotlib 3.3.4 -> 3.11.2; unpinned SciencePlots -> 2.2.2. Added explicit pins for existing utility dependencies: seaborn 0.13.2, openpyxl 3.1.5, PyAutoGUI 0.9.54 and keyboard 0.13.5. Installer/build tools are pip 26.2.1, setuptools 84.0.0 and wheel 0.48.0.

Import compatibility modifications were limited to `if __name__ == "__main__":` guards around existing executable bodies in `count parameters.py`, `data analysis/sample count.py`, `plotter/Figure 2.py`, `plotter/Figure 4a.py`, `plotter/Figure 4b.py`, `plotter/Figure 5.py` and `plotter/color.py`. Previously importing these utilities could parse arguments, read experiment data or produce plots; now these actions occur only when the files are executed directly. Original statements and line endings are preserved. An AST comparison against the original commit confirmed that removing just the guards recovers the original code.

Validation: all 28 original Python files plus the verifier imported successfully (29/29), all ten direct third-party imports passed, `pip check` found no broken requirements, and the installed distributions exactly match the 45-package lock. Reports are in `environment_checks/`. SHA-256 checks confirmed `Model/Model.py`, `Model/Compare_Models.py`, `dataloader/dataloader.py` and `utils/util.py` are unchanged; all main training scripts are also unchanged.

Expected effect: guards change import-time behavior only. Dependency upgrades may alter numerical behavior, performance and later plotting/analysis compatibility even though the mathematical model is unchanged; published accuracy and bitwise equivalence have not been verified. No dataset split, feature, label, loss, metric implementation or checkpoint behavior was changed. No training, optimizer updates or dataset evaluation were run. Existing scientific/evaluation issues above remain open, and a bounded training smoke test is still required before a long experiment.

### Smoke-test status update

The required bounded smoke test was subsequently completed using `scripts/smoke_test.py`: one existing MIT split, one model, and exactly three epochs. The original model, dataloader, main scripts, features, labels, split rules and loss equations remained unchanged. The complete environment/runtime record and preliminary MAE, MAPE, RMSE and MSE are in `ENVIRONMENT.md`; the machine-readable run record is under `results/smoke_tests/mit_20260914T055041Z/run_manifest.json`. A pandas 2.3.3 dtype-assignment warning was observed and documented; it did not require a source fix. No long training has been run.

### First full MIT experiment instrumentation, 17 September 2026

Added `scripts/reproduce_mit_one.py`, a single-experiment runner using the original `PINN.Train`, original MIT loader, and current `main_MIT.get_args` defaults. A subclass records the return values of the original epoch/validation/test methods; it adds no training forwards, backwards, validation shuffles, or test passes during optimization. It records per-epoch losses, used/next learning rates, validation MSE, RNG states, ordered files, tensor hashes, source hashes, timing and environment configuration under `results/reproduction/MIT/run_01/`. No global seed is introduced.

To satisfy the requested best checkpoint, the runner immediately serializes `best_checkpoint.pth` at each validation improvement; the original deferred `model.pth` is retained separately. After training it reloads the best weights and verifies the resulting predictions against the saved best-epoch arrays. Standard MAPE is calculated with ground truth as denominator, alongside a separately labeled author-convention MAPE. These are artifact/evaluation instrumentation changes, not compatibility fixes or changes to equations, optimization or data protocol.

The historical MIT ZIP logs record alpha=1 and beta=50, unlike the current script's alpha=0.5 and beta=0.01. This run follows the explicitly requested current-script plan. The historical archive is a reference with a known configuration mismatch, not evidence that this is a matching historical training configuration.
