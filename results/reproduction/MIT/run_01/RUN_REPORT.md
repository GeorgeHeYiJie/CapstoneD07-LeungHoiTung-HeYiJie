# MIT reproduction: run 01

Completed 17 September 2026. Exactly one experiment was run with the current MIT script defaults documented in `docs/FULL_REPRODUCTION_PLAN.md`. No remaining experiment was launched.

## Outcome

| Item | Result |
| --- | ---: |
| Maximum configured epochs | 200 |
| Best validation epoch | **65** |
| Stopping epoch | **86** |
| Best validation MSE | 0.0000685328050 |
| Training-loop runtime | 207.83 seconds |
| Total runner runtime, including loading, verification and plots | **218.76 seconds (3 min 39 sec)** |
| Test samples | 14,170 |
| Checkpoint reload prediction difference | **0.0** |

The original early-stopping rule stopped training after 21 consecutive epochs without improving on epoch 65. All loss and prediction values were finite. Training showed a large loss spike at epoch 68 and then recovered partially, but validation had not regained its best value by epoch 86. The saved best checkpoint predates that spike.

## Best-validation checkpoint test metrics

These are sample-weighted metrics over all 14,170 held-out test pairs. SOH is represented as a fraction; MAE and RMSE therefore use fractional SOH units, and MSE uses squared fractional units. MAPE uses ground truth as the denominator. There were no zero-valued targets.

| Metric | Run 01 | Authors' archived ten-run mean | Archived run range |
| --- | ---: | ---: | ---: |
| MAE | **0.00656423** | 0.00601779 | 0.00574993–0.00652520 |
| MAPE (%) | **0.692755** | 0.635743 | 0.607661–0.689232 |
| RMSE | **0.00909792** | 0.00827617 | 0.00788037–0.00907445 |
| MSE | **0.0000827721** | 0.0000686079 | 0.0000621003–0.0000823456 |

The MAE corresponds to about **0.656 SOH percentage points**; RMSE corresponds to **0.910 percentage points**. Run 01 has slightly higher errors than the worst of the ten archived runs. MAPE is 0.0570 percentage points above the archived mean; RMSE is approximately 9.9% higher than the archived mean.

The original logger's reversed-argument MAPE is retained in `logging.txt`. For this checkpoint it is 0.0069348235 as a fraction (0.693482%). It is labeled separately in `metrics.json`; it is not the standard MAPE in the table above.

## Comparison provenance and limits

The reference above was recomputed directly from the ten `true_label.npy`/`pred_label.npy` pairs inside the unchanged repository archive `results/Ours/MIT results.zip`, using the same standard metric definitions as this run. All archived runs contain 14,170 test samples. The ordered training and test battery lists in archive Experiment1 match this run, and its stored test targets match this run bit for bit.

**A configuration discrepancy prevents calling this an exact reproduction of the historical result:** all ten bundled MIT logs record `alpha=1, beta=50`, while the current script and requested plan use `alpha=0.5, beta=0.01`. The run retained the requested settings. The historical logs also have different training-loss logging contents, so their old implementation cannot be assumed identical to the current source. This comparison is a repository-reference comparison, not a confirmed match to the paper's Table 2 aggregation. The analysis script in the repository additionally uses inferred battery boundaries and battery-level averages, which differ from the sample-weighted calculation above.

The paper is [Wang et al., Nature Communications (2024)](https://doi.org/10.1038/s41467-024-48779-z). Its publisher table could not be reliably retrieved during this run; no unverified paper-specific MIT table value is substituted here. The repository's actual saved author predictions provide the directly auditable numerical reference requested.

## Settings and preservation

- Original batch size 512, original two Adam optimizers, original 30-epoch warm-up/cosine scheduler, and original post-epoch scheduler timing.
- Original loss weights alpha=0.5 and beta=0.01.
- Original MIT split: battery IDs divisible by five held out; pooled remaining pairs split 80/20 with random_state=420; directory order not sorted.
- 54,053 training pairs, 13,514 validation pairs and 14,170 test pairs.
- No global seed set, matching the author script; initial RNG states saved for audit.
- Original model, dataloader, main MIT script and metric utility unchanged. The inherited original `PINN.Train` executed the entire training/stopping loop.
- Environment: Python 3.12.14, PyTorch 2.14.0+cpu, 8 intra-op threads and 8 inter-op threads. This is the compatibility environment, not the historical dependency stack.

The separate runner records original method returns. It immediately saves `best_checkpoint.pth` on validation improvement to meet the requested best-checkpoint requirement. This is an explicit artifact-saving addition: it does not modify the training equations, selection criterion, optimizer updates or RNG seeding. The original `model.pth` is also retained, but its deferred shallow-state behavior means it should not be treated as the best-epoch checkpoint. Loading `best_checkpoint.pth` and evaluating with the original `PINN.Test` reproduced the saved prediction arrays exactly.

Training losses in `loss_history.csv` follow the original equal-weight average of minibatch losses, including the shorter final batch. The physics component is the original batch-summed penalty before beta weighting. Validation history is the original full-loader SOH MSE, not a validation PDE/physics objective.

## Artifacts

- `loss_history.csv`: all 86 epochs, data/PDE/physics/total training losses, validation MSE, used/next learning rates and epoch training durations.
- `loss_history.png`: training loss components and validation MSE.
- `best_checkpoint.pth`: actual best-validation weights, epoch and validation MSE.
- `model.pth`: unmodified author-loop checkpoint artifact, retained for traceability.
- `true_label.npy`, `pred_label.npy`: best-validation epoch test labels and predictions.
- `predicted_vs_true_SOH.png`: all held-out test predictions against measured SOH.
- `metrics.json`: final metrics, best/stopping epochs, runtime and checkpoint verification.
- `test_history.json`: test results at each validation improvement, preserving the author's evaluation timing.
- `run_configuration.json`: arguments, device, source commit/hashes, ordered split manifest and effective tensor hashes.
- `rng_initial.pth`: initial Python, NumPy and PyTorch RNG states (contains Python objects; only load this locally generated trusted file).
- `dataset_sha256.json`: SHA-256 hashes of every MIT input file.
- `source_diff.patch`, `source_status.txt`, `runner_source.py`, `requirements-reproduction.txt`: source and environment records.
- `preflight_import_report.json`: successful environment import/dependency verification.
- `console.log`, `logging.txt`: full console output and original logger output.
- `author_archive_comparison.json`: archived per-run metrics, aggregate statistics, configuration evidence and matching-data checks.

## Runtime warnings

The known pandas dtype-assignment FutureWarning recurred; loading succeeded with the pinned version. Matplotlib could not write its default external font cache and emitted an ignored cleanup error at process exit. Both requested plots were successfully saved, opened and visually inspected. Neither warning changed model inputs or training. No compatibility code fix was applied during the run.

## Is this valid enough to continue?

**Yes as a valid first run of the current-code protocol.** The environment passed checks, expected splits/counts were confirmed, all losses were finite, early stopping behaved correctly, and the best checkpoint exactly reproduces its saved predictions. The errors are close to the repository reference, though slightly above its ten-run range.

**Do not describe a continuation with these settings as an exact historical/paper reproduction yet.** Resolve and explicitly document the alpha/beta discrepancy before committing the remaining nine runs to that claim. The epoch-68 spike also means this single run does not establish stable convergence; repetition is needed to measure variability. No test-based setting changes were made, and no additional experiments were started.
