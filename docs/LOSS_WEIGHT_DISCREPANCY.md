# MIT loss-weight discrepancy investigation

Investigated 17 September 2026 against local HEAD `cc3cc5053caf38f04e0665f7f88cb109144d035e`. No training, optimizer updates, model changes or configuration changes were made. Checks included the complete available local Git history (not a shallow clone), all ten archived MIT logs and prediction sets, README, `fix.md`, the current and historical source, the published equations, and official supplementary information. Loading the existing dataset to calculate a label-only statistic did not train or instantiate a model.

## Findings and recommendation

**Recommend reproducing the paper/archive configuration for the FYP's first objective, rather than spending the remaining nine runs on current repository behavior.** MIT's published weights are **alpha=1, beta=50**. All ten archived logs agree. The current values **0.5, 0.01** were introduced in September 2026, after publication.

However, this is not just a hyperparameter discrepancy. The physics loss changed twice, and the original released expression disagrees with the paper. Therefore **do not simply put beta=50 into the current model and call it a paper reproduction**. First distinguish an archival numerical-reproduction baseline from a separate implementation of the paper's intended equation. This document recommends the archival baseline first, with its defect disclosed, because the immediate objective is reproducing the reported results.

Run 01 remains a valid current-repository experiment. Keep it intact, but do not pool it with nine differently configured historical runs to form a ten-run paper-reproduction mean.

## 1. Which weights belong to the paper's MIT results?

**Confirmed: alpha=1 and beta=50.**

The official [Supplementary Information, Supplementary Note 3, printed page 14](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-48779-z/MediaObjects/41467_2024_48779_MOESM1_ESM.pdf#page=14) explicitly specifies these weights for MIT and TJU, and batch size 512 for MIT. This is direct publication evidence, not an inference from today's defaults.

The [paper's Methods, equations (8) and (9)](https://doi.org/10.1038/s41467-024-48779-z) define a predicted-SOH monotonicity penalty and its weighted addition to data and PDE losses. In minibatch notation, the intended monotonicity term is:

```text
L_mono = sum ReLU(u2 - u1)
L_total = L_data + alpha*L_PDE + beta*L_mono
```

Here `u1` and `u2` are predicted SOH at successive cycles. The paper's written equation contains no multiplication by measured `y1-y2`.

[`README.md`](../README.md), section 3, identifies the bundled result and analysis files as corresponding to the manuscript. That statement is independently supported by the archive-to-supplement check below.

## 2. Were the comparison archives generated with 1 and 50?

**Yes, according to every archived run's own recorded configuration.**

Examined `logging.txt`, `true_label.npy` and `pred_label.npy` for Experiment1 through Experiment10 inside [`results/Ours/MIT results.zip`](../results/Ours/MIT%20results.zip). Every log records `alpha:1` and `beta:50`. All recorded arguments other than the per-experiment output directory agree across the ten logs. Logs are dated 7 July 2023. The ZIP was added to the public repository in commit `a70f90f` on 23 April 2024; the July 2023 training executable/commit is not embedded in the archive.

The earlier comparison was recomputed from exactly these ten saved prediction sets. Its means are:

| Metric | Archived ten-run mean used in run-01 comparison |
| --- | ---: |
| MAE | 0.00601778594 |
| MAPE, standard ground-truth denominator | 0.635742797% |
| RMSE | 0.00827617325 |
| MSE | 0.0000686079075 |

Each run has 14,170 samples. These are whole-test-set, sample-weighted metrics calculated separately per run and then averaged across runs. See the preserved [comparison JSON](../results/reproduction/MIT/run_01/author_archive_comparison.json).

### Direct link to published supplementary results

I independently applied the existing MIT analysis procedure to the archive arrays: infer battery boundaries using upward target jumps greater than 0.05, apply its existing slicing rules, use its reversed-argument MAPE convention, and average each battery's metrics across ten runs. **All 23 MIT battery rows reproduce the printed MAPE and RMSE values in Supplementary Table S.6 to four decimal places.** This establishes a numerical link between this archive and the published MIT supplementary results.

That historical analysis yields an equal-weight battery/run mean MAPE of approximately **0.667321%** and RMSE **0.00751015**. These differ from the sample-weighted standard metrics above because of battery weighting, denominator convention, and the analysis script's boundary slicing. The earlier 0.635743% comparison is a corrected recalculation of the archived predictions, not a verbatim paper table entry. Do not mix these aggregations when reporting success.

## 3. When did the current defaults appear?

The local history resolves the sequence. Dates below are recorded Git author dates.

| Date / commit | MIT weights | Relevant change |
| --- | --- | --- |
| 23 April 2024, `5d57a9d` | Model source upload | Released model uses `ReLU(y2-y1).sum()` for physics loss. |
| 23 April 2024, `d72414a` | **1, 50** | MIT entry point uploaded; its default early-stop argument is 10, unlike the archived logs' 20. |
| 21 May 2024 | **1, 50 in supplement** | Paper published; equation (8) uses predicted SOH differences. |
| 11 July 2024, `1e94ec8` | 1, 50 | Changes physics loss from measured-label differences to `ReLU(u2-u1).sum()`. |
| 16 October 2024, `7d6fe13` | Model change | Changes physics loss to `ReLU((u2-u1)*(y1-y2)).sum()`, allowing measured capacity regeneration. |
| 16 October 2024, `9de69bc` | **1, 0.02** | Reduces MIT beta from 50 to 0.02. |
| 2 September 2026, `cc3cc50` | **0.5, 0.01** | Changes both MIT weights, early_stop from 10 to 20, output location, adds grid-search helper, activates `main()` in the executable guard. |

Thus the current pair was introduced later, and there was an intermediate pair. The current model formula dates from October 2024; the September 2026 commit changes MIT settings but does not change `Model/Model.py` itself.

[`fix.md`](../fix.md), added with the September 2026 commit, explicitly explains in Chinese that bugs in the initial code were fixed, causing alpha and beta to differ from the supplementary file. This author's note supports using historical versions to investigate the discrepancy. Its broad statement that other code is unchanged should not replace inspection of the earlier formula changes visible in Git.

Useful audit commands (read-only):

```powershell
git log --all --format='%h %aI %s' -- main_MIT.py Model/Model.py
git show 1e94ec8 -- Model/Model.py
git show 7d6fe13 -- Model/Model.py
git show 9de69bc -- main_MIT.py
git show cc3cc50 -- main_MIT.py fix.md
git diff d72414a HEAD -- Model/Model.py dataloader/dataloader.py utils/util.py
```

## 4. Other parameter and implementation differences

### Recorded arguments: archive versus current MIT defaults

All ten logs were parsed and compared with the current argument defaults, rather than inspecting Experiment1 alone.

| Setting | Archived ten runs | Current main_MIT.py |
| --- | --- | --- |
| alpha / beta | **1 / 50** | **0.5 / 0.01** |
| Maximum epochs | 200 | 200 |
| Batch size | 512 | 512 |
| early_stop | **20** | **20** |
| Normalization | min-max | min-max |
| Warm-up epochs | 30 | 30 |
| Warm-up / base / final LR | 0.002 / 0.01 / 0.0002 | Same |
| Dynamics LR | 0.001 | 0.001 |
| u_layers_num / u_hidden_dim | 3 / 60 | Same (actual Solution_u architecture is hard-coded) |
| F_layers_num / F_hidden_dim | 3 / 60 | Same |
| Number of experiments | 10 archived experiments | `range(10)` |
| Log filename | logging.txt | logging.txt |
| Effective output location | results/MIT results/ExperimentN | 202608/Push/MIT results/ExperimentN |

**No other recorded numerical argument differs between the archive and current script.** The older script's default early_stop=10 is not the archive's actual setting; an archival reconstruction must override it to 20. Output destinations differ but do not define the mathematical model.

### Physics-loss changes are scientifically material

| Version | Implemented physics term | Effect on learning |
| --- | --- | --- |
| Initial public code, consistent with archived logs | `sum ReLU(y2-y1)` | Depends only on measured labels. Its derivative with respect to network weights is zero. It adds to logged total loss without providing a physics gradient. |
| July 2024 fix; matches paper's monotonicity equation | `sum ReLU(u2-u1)` | Penalizes predicted SOH increases. |
| October 2024 through current HEAD | `sum ReLU((u2-u1)*(y1-y2))` | Penalizes predictions that disagree with measured local direction; magnitude is scaled by the measured SOH change. |

There is no universal conversion factor between these beta values. In particular, multiplying the present label-weighted expression by 50 produces a fourth experimental combination, not the archived implementation or the paper's written equation.

### Evidence about the archived formula

Every epoch-level `[Train]` log entry in all ten archives reports physics loss **0.043094**, independent of epoch and initialization. Computing the initial released label-only penalty on the current unchanged MIT training tensors gives:

```text
sum over training pairs of ReLU(y2-y1) / number of training minibatches
= 0.043093645347739164
```

There are 106 minibatches. This reproduces the log's six-decimal value exactly. Individual minibatch values vary with shuffling; the epoch average of batch sums is fixed. This is strong evidence that the archive used the initial label-only loss or an equivalent implementation. It also explains why the logged total can stay near 2.15 even after SOH errors become small: 50 times this fixed component is approximately 2.15468.

**Evidence limit:** the archived runs do not include a source revision or executable. Logs alone cannot exclude a logging-only error in an otherwise different training program. The matching released source and numerical constant support an archival reconstruction, but do not prove the exact July 2023 executable beyond doubt. The original released defect must not be represented as the paper's intended physical constraint.

### Other behavior and environment

- Comparing `d72414a` with HEAD shows no changes to MIT feature CSVs, `dataloader/dataloader.py`, or `utils/util.py`. The original data/PDE equations, architecture, Adam construction, scheduler and training loop in `Model/Model.py` are unchanged apart from the physics expression across these revisions.
- Archived Experiment1's ordered battery lists and target array match run 01, as checked previously. The split remains 102 train/validation batteries and 23 held-out test batteries, with pair-level random_state=420. There is no declared global training seed in the source or archive configs.
- The current and historical source use mean-reduced paired data/PDE losses and a batch-summed physics loss. The paper writes summation-form objectives; weights cannot be transferred between mean and sum conventions without considering scaling.
- The older analysis code expected five metric outputs including R2, whereas the supplied helper returns four; September 2026 updates reconcile the call sites. Reversed MAPE argument order and heuristic battery slicing remain relevant to matching published summaries.
- The source retains the deferred shallow `state_dict` checkpoint issue. The dedicated best checkpoint saved for run 01 is a documented artifact fix; it does not establish that archived `model.pth` files equal their saved best-epoch prediction arrays.
- README specifies Python 3.7.10/PyTorch 1.7.1. The supplement identifies an Intel i5-10400F CPU for its implementation/inference context. Our environment is Python 3.12.14/PyTorch 2.14.0+cpu on Ryzen 7 5800H. Archive logs do not capture full runtime versions, RNG states or Adam internals, so bitwise reproducibility is not established.

## 5. Configuration for a faithful reproduction

The evidence requires distinguishing two historical targets:

### Recommended first target: archived numerical results

Use alpha=1, beta=50 and **archive-recorded early_stop=20**, with the other archived arguments in section 4. Pin an isolated publication-era source snapshot, such as `d72414a`, rather than editing the current baseline. Retain its actual label-only physics expression and label the resulting study explicitly as an **archival implementation reproduction with an inactive physics-loss gradient**. Preserve source and data hashes, the existing split, normalization, optimizer and scheduler. Record modern dependency differences if the current compatibility environment is used.

This is the most defensible reconstruction for matching the archived numerical reference. The old script's inactive main guard, output directory and early_stop default require an explicit separate runner; these must be documented, not silently patched. A bounded smoke test of that historical path should precede any full historical run. None of those changes or runs was performed in this investigation.

Evaluate with both the historical aggregation/metric conventions for matching published tables and correctly defined standard metrics for engineering interpretation. Do not silently fix the historical formula, MAPE, or sample selection while calling the outcome an exact archival reproduction.

### Separate target: the paper's intended mathematical model

Use alpha=1, beta=50 with prediction-based `sum ReLU(u2-u1)`, as in equation (8) and the July 2024 fix `1e94ec8`. Keep archive-recorded settings such as early_stop=20 explicitly controlled. Label this as a **paper-equation reproduction / corrected monotonicity experiment**, not an exact rerun of the archived implementation. Matching its numerical results to the archive is not guaranteed because it adds an active gradient that the initial source lacks.

The present regeneration-aware expression with alpha=0.5, beta=0.01 is a third, later protocol. It is useful as the maintained-code baseline, but it should not be substituted for either historical target. Run 01 belongs to this third protocol.

## Decision

**Choose: reproduce the paper/archive configuration.** The supplement directly supports 1/50, the ten archived logs independently confirm it, and the archive predictions reproduce Supplementary Table S.6. For the immediate goal of reproducing published numbers, start with the archival implementation reconstruction described above, disclose its label-only-loss defect, and then test the paper-equation version separately. Continuing nine current-code runs would quantify the variability of a later algorithm rather than resolve the publication-reproduction question.

No new training was run and no existing model, data, loss equation, split, or completed result was modified.
