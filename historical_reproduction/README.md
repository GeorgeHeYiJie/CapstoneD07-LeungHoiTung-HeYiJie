# Isolated historical MIT reproduction

This directory preserves the label-only physics-loss defect of the publication-era code for archival numerical reproduction. It is not the current PINN baseline and is not an implementation of the paper's intended prediction-based monotonicity equation.

`source/` contains byte-for-byte Git blobs from commit `d72414ac8799747db784203b4c1826ca118dc23a` (23 April 2024). `source_manifest.json` records their SHA-256 hashes. Do not edit those snapshot files.

`run_mit.py` imports the snapshot modules ahead of the current repository modules, checks actual import paths and hashes, and invokes the historical inherited `PINN.Train`. It supplies archive-recorded `early_stop=20` instead of the historical script default 10. The weights are the historical defaults alpha=1 and beta=50. No global seed is introduced.

Exactly one experiment is supported per invocation. The full output directory is fixed to `results/reproduction/MIT/historical_run_01/`, and an existing directory causes a refusal to overwrite. The `--smoke` flag uses three epochs and the separate `results/smoke_tests/historical_mit_01/` output. Both were authorized for the initial reconstruction; neither command resumes or silently starts another run.

From the repository root, using the established compatibility environment:

```powershell
.\.venv-reproduction\Scripts\python.exe historical_reproduction/run_mit.py --smoke
.\.venv-reproduction\Scripts\python.exe historical_reproduction/run_mit.py
```

See [HISTORICAL_REPRODUCTION.md](../docs/HISTORICAL_REPRODUCTION.md) for provenance, the defect, exact settings, results and limitations. Current model files and all existing Run 01 artifacts are protected by the pre-run hashes in `protected_before.json`.
