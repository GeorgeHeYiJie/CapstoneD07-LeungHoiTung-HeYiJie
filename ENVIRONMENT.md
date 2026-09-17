# Reproduction environment

This is a **modern compatibility environment**, not the authors' original software stack. It targets Windows x86-64, Python **3.12.14**, and PyTorch **2.14.0+cpu**. It was prepared for repository commit `cc3cc5053caf38f04e0665f7f88cb109144d035e` on 14 September 2026. The mathematical model, losses, optimizer code, data splits, features, labels and preprocessing are unchanged.

## Why the original versions were not used

The README specifies Python 3.7.10, torch 1.7.1, scikit-learn 0.24.2, NumPy 1.20.3, pandas 1.3.5 and Matplotlib 3.3.4. SciencePlots is listed without a version.

The machine inspection found:

- `python` on PATH resolves to a WindowsApps alias that fails to launch. A registry entry exists for Microsoft Store Python 3.10; it is not the authors' Python 3.7 environment.
- No `conda`, `py`, or `uv` executable was found on PATH, and no Python 3.7 installation was found in the inspected installation locations/registry entries.
- A working bundled CPython **3.12.14 (64-bit AMD64)** is available. It was used to create the project virtual environment with system site-packages disabled.
- A pip binary-only dry-run for the original packages fails with `No matching distribution found for torch==1.7.1` on Python 3.12.
- Official PyPI release metadata confirms that none of the five pinned scientific packages has a CPython 3.12 Windows x64 wheel. Torch 1.7.1, NumPy 1.20.3, scikit-learn 0.24.2 and Matplotlib 3.3.4 have Windows wheels through Python 3.9; pandas 1.3.5 also has Python 3.10 wheels.

Therefore the original stack cannot be installed unchanged into the available working Python environment using its published binaries. This does not establish that the historical stack could never run in a separate legacy Python 3.7 installation; such an installation/source build was not attempted. Python 3.7 is end-of-life; Python 3.12 remains supported. See the [Python version status](https://devguide.python.org/versions/) and [original torch release files](https://pypi.org/project/torch/1.7.1/#files).

The local evidence for the original wheel check is saved in `environment_checks/original_compatibility.json`.

## Changes from the authors' dependencies

| Component | Authors' environment | This environment | Reason and compatibility implications |
| --- | --- | --- | --- |
| Python | 3.7.10 | 3.12.14 | Working supported interpreter available on this machine; enables current binary wheels. |
| PyTorch | 1.7.1 | 2.14.0+cpu | Current stable PyPI release at setup time with a Windows/Python 3.12 CPU wheel. Same network and autograd expressions; numerical equivalence to 1.7.1 is not claimed. |
| NumPy | 1.20.3 | 2.5.3 | Python 3.12 wheel; major-version numerical/API differences need validation in later reproduction runs. |
| scikit-learn | 0.24.2 | 1.9.1 | Python 3.12 wheel; split and metric call sites remain unchanged. |
| pandas | 1.3.5 | 2.3.3 | Python 3.12 wheel. Deliberately retains the 2.x line rather than adopting pandas 3.x's additional behavior changes during initial reproduction. This does not eliminate older API deprecations. |
| Matplotlib | 3.3.4 | 3.11.2 | Python 3.12 wheel for plotting imports. |
| SciencePlots | Unpinned | 2.2.2 | Explicitly pinned for reproducible style imports. |
| seaborn | Not listed; imported by figure scripts | 0.13.2 | Added to cover all repository imports. |
| openpyxl | Not listed; explicitly used for Excel reads | 3.1.5 | Added for the existing spreadsheet analysis/plotting workflows. |
| PyAutoGUI | Not listed; imported by `plotter/color.py` | 0.9.54 | Added so the optional color utility imports; no mouse/keyboard automation was performed. |
| keyboard | Not listed; imported by `plotter/color.py` | 0.13.5 | Added to cover that utility's imports; no hooks were installed by the verification script. |
| pip | Unspecified | 26.2.1 | Pinned installer used for this environment. |
| setuptools | Unspecified | 84.0.0 | Pinned packaging/build dependency, also required by current PyTorch. |
| wheel | Unspecified | 0.48.0 | Pinned support for building the small pure-Python utility packages supplied as source archives. |

`requirements-reproduction.txt` pins **all installed distributions**, including transitive dependencies. The authors did not supply a transitive dependency lock, so unknown original transitive versions cannot be reconstructed or compared honestly. Every transitive dependency in the lock is newly recorded, rather than asserted to be an upgrade from a known original version. They support tensor operations, scientific routines, plotting, date/time handling, Excel files, or the optional GUI utility. No torchvision or torchaudio dependency was added because the repository does not import them.

The [PyTorch package metadata](https://pypi.org/project/torch/2.14.0/) supports Python 3.12; the CPU wheel comes from the [official PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/). The installation procedure follows the [PyTorch installation documentation](https://pytorch.org/get-started/locally/).

## CPU and GPU choice

The installed PyTorch build is explicitly CPU-only. The machine has an NVIDIA GeForce RTX 3070 Laptop GPU with driver 616.56, but CUDA packages are not required for import verification. `torch.version.cuda` is `None` and `torch.cuda.is_available()` is `False` in this environment by design.

A CUDA reproduction environment can be created separately later and should get its own dependency lock and smoke test. Do not silently replace the CPU build or claim GPU training was validated. CPU/GPU kernels and dependency upgrades can change floating-point results even with identical equations.

## Use the existing environment

From the repository root in PowerShell:

```powershell
.\.venv-reproduction\Scripts\python.exe --version
.\.venv-reproduction\Scripts\python.exe -m pip check
.\.venv-reproduction\Scripts\python.exe verify_environment.py
```

Activation is optional. Calling the executable directly avoids the broken WindowsApps alias and PowerShell activation-policy issues. For an editor, select `.venv-reproduction\Scripts\python.exe` as the interpreter.

## Recreate the environment

Use a fresh checkout/directory with Windows x86-64 and the same Python patch version for the closest match. Do not copy an existing virtual environment between machines. Set the base executable to an installed **Python 3.12.14** interpreter; the bootstrap path used on this machine is shown below.

```powershell
$basePython = 'C:\Users\oscar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $basePython -c "import sys; assert sys.version_info[:3] == (3,12,14); print(sys.version)"
& $basePython -m venv .venv-reproduction
.\.venv-reproduction\Scripts\python.exe -m pip install pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 packaging==26.3
.\.venv-reproduction\Scripts\python.exe -m pip install --no-build-isolation -r requirements-reproduction.txt
.\.venv-reproduction\Scripts\python.exe -m pip check
.\.venv-reproduction\Scripts\python.exe verify_environment.py
```

On another machine replace `$basePython` with its real interpreter path. The current environment references the bundled interpreter; if that bundle is removed or relocated, recreate the environment from a standalone interpreter rather than editing `pyvenv.cfg` manually.

The requirements file selects PyPI plus the official CPU index and pins the `+cpu` build. Build isolation is disabled in the recreation command so utility source distributions use the explicitly installed build tools. The initial installation resolved dependencies once, after which all installed versions were locked. The lock fixes package versions; it is not a hash-verified, offline wheel archive or a guarantee of bitwise-identical training. It is specifically validated for Windows x86-64/Python 3.12, not every operating system.

During setup the sandbox blocked pip's temporary wheel files, so pip bootstrapping and downloads were completed through approved elevated tool calls. All project dependencies were installed into `.venv-reproduction`, not the shared bundled Python or a system environment. Standard pip download/build caches may be outside the repository.

## Source compatibility changes

Seven utility files had unguarded top-level executable bodies. Imports could parse command-line arguments, instantiate models, read result files or create plots. These bodies were placed under `if __name__ == "__main__":`:

1. `count parameters.py`: defer argument parsing, model construction and parameter-count printing.
2. `data analysis/sample count.py`: defer dataset file traversal and counting.
3. `plotter/Figure 2.py`: defer dataset reads and plotting.
4. `plotter/Figure 4a.py`: defer prediction reads and plotting.
5. `plotter/Figure 4b.py`: defer result-spreadsheet reads and plotting.
6. `plotter/Figure 5.py`: defer small-sample spreadsheet reads and plotting.
7. `plotter/color.py`: defer its example calculation/print; functions and dependencies remain importable.

When these files are run directly, their existing statements execute in their original order and scope. Importing them now defines their contents without running those workflows. An AST comparison against `git show HEAD:<file>` verified that removing only the new guard recovers the original program statements. Original line endings were retained.

The following files were left byte-for-byte unchanged (SHA-256):

| File | SHA-256 |
| --- | --- |
| `Model/Model.py` | `b146b179a7267a50bed231a99ad5b161b14f8c527b62c376bf286aa7995e0fca` |
| `Model/Compare_Models.py` | `d2758711394285bce9a7f6722d3504fc1cfa067942fda059c5e67f2437bbb812` |
| `dataloader/dataloader.py` | `192169eed4e172934d39ba4773074abd37f147170c1996327d6fd0ad2a8790fc` |
| `utils/util.py` | `db7a7e31e785ab2a046bee2e2c20a595161e900f1dffe40c390a5084ff8dd9f2` |

All main training scripts are unchanged. Existing scientific/evaluation issues, including MAPE argument order, checkpoint snapshots and TJU cross-batch indexing, are documented in `REPRODUCTION_NOTES.md` and were not silently fixed during environment setup.

## Verification scope

Completed successfully on 14 September 2026:

- **29/29 source files imported**: all 28 original repository Python files plus `verify_environment.py`.
- **10/10 direct dependency imports passed**, including optional plotting and color utilities.
- **`pip check` passed**: `No broken requirements found.`
- The requirements lock exactly matches **all 45 installed distributions**, with no missing, unpinned or extra installed distributions.
- The four recorded model/data/metric source hashes match their pre-setup values.
- All seven guarded utility bodies match the original source AST after removing only the new `__main__` guard.

`verify_environment.py` imports all ten explicitly used third-party packages and discovers every repository `.py` file, excluding virtual environments, caches and generated environment reports. It actually imports each source file in a fresh subprocess using `importlib`, including filenames containing spaces and hyphens; it does not skip arbitrary statements using an AST filter. Each import has a timeout. The verifier itself is also checked.

Verification uses the noninteractive Matplotlib `Agg` backend and a project-local configuration cache. Styles still import normally; figures are not rendered. The verifier records versions, source hashes, CUDA availability and per-file output in `environment_checks/import_report.json`, and runs `pip check`. Generated caches/reports and the virtual environment are excluded by `.gitignore`.

Import verification does not establish that every plotting or result-analysis workflow runs end-to-end. Existing missing-path assumptions, old `ExcelWriter.save()` calls, TeX rendering needs and fine-tuning result-parser issues still require separate checks when those workflows are requested. Python 3.12 can also warn about the authors' non-raw strings containing backslashes in plot labels; an import warning is not an import failure.

No full training, optimizer updates, dataset evaluation or reproduction metrics are part of this environment task. MAE, MAPE, RMSE and MSE will be reported together when an actual model evaluation is performed. A bounded data/training smoke test is still required before any long training run.

## MIT three-epoch smoke test, 14 September 2026

`scripts/smoke_test.py` was added as a separate bounded runner. It imports the unchanged `main_MIT.load_MIT_data` function and the unchanged `Model.Model.PINN` class. It performs one MIT split and one experiment for exactly three epochs, then evaluates the final-epoch model on the existing held-out MIT test batteries. It does not call or modify any original `main()` function.

The run passed on CPU with training tensors shaped `(54053, 17)` and targets shaped `(54053, 1)`. It confirmed finite inputs, targets and test predictions, completed optimizer updates, printed all three loss components and their weighted total separately, evaluated 14,170 test pairs, and wrote a final-epoch checkpoint, predictions, labels and a run manifest under `results/smoke_tests/mit_20260914T055041Z`. These preliminary sample-weighted metrics use SOH as a fraction and standard ground-truth-denominator MAPE: MAE 0.0137295770, MAPE 1.463705%, RMSE 0.0183147759 and MSE 0.0003354310. They are smoke-test results, not a claim of published-result reproduction.

Modern pandas 2.3.3 emitted a `FutureWarning` while the unchanged loader assigned normalized floating-point values back into the integer-typed generated `cycle index` column. The assignment completed and produced a float32 model input after conversion. No source compatibility fix was applied because changing the loader was unnecessary for this run and could affect preprocessing behavior. This warning may become an error in a future pandas release; pandas remains pinned at 2.3.3 for the reproduction environment.

No mathematical model, split rule, feature, label, normalization formula or loss equation was changed. The smoke runner sets Python, NumPy and PyTorch seeds to 420 for this experiment, records the unsorted file order used by the existing MIT split, and calculates standard evaluation metrics outside the authors' reversed-argument MAPE helper. That corrected MAPE is explicitly labeled and does not replace historical values emitted by the authors' code.

## First full MIT experiment, 17 September 2026

The unchanged compatibility environment passed `verify_environment.py` and `pip check` before one full experiment. `scripts/reproduce_mit_one.py` recorded the inherited original training loop under `results/reproduction/MIT/run_01/`. Training stopped normally at epoch 86, with best validation at epoch 65; the immediate best-checkpoint snapshot reproduced saved predictions exactly. Total runner runtime was 218.76 seconds. See the run's `RUN_REPORT.md` and `metrics.json`.

The existing pandas dtype-assignment warning recurred without failure. Matplotlib could not write its default external font cache and emitted an ignored temporary-cache cleanup error at exit, but both plots were saved and visually checked. No compatibility modification was required or applied to the original code. A future plotting invocation can set `MPLCONFIGDIR` to a writable project-local directory. This run's extra immediate best-checkpoint snapshot and standard-MAPE post-processing are documented artifact/evaluation changes, not mathematical-model or compatibility changes.
