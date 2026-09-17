# Simplified PIMOE development plan

Prepared 17 September 2026. **Proposal only: no new model implementation or training is authorized by this document.**

## 1. Project decision and scope

The reproduction stage is sufficiently complete for FYP progression, as confirmed by the student and mentor. Treat the existing PINN reproduction as the reference baseline. Preserve all reproduction results and documents unchanged. Do not run historical experiments 02–10 or rebuild the historical environment unless explicitly requested later.

This decision supersedes the *future-work recommendations* in the historical audit; the audit remains an unchanged record of that earlier stage. Exact archive matching is no longer a prerequisite for model development.

The project goal in `AGENTS.md` is to reproduce PINN4SOH and then develop a simplified physics-informed mixture of experts for battery degradation prediction. The available project description is that goal plus the student's conversation instructions. No separate formal project brief or identifiable PIMOE paper was found in the local document inventory. Accordingly, this plan proposes our own small **physics-regularized MoE for same-cycle SOH estimation**. It does not claim to reproduce an unspecified published PIMOE architecture.

Research question: **Can a small mixture of experts, guided by charging features and a soft degradation-trend penalty, give a useful accuracy/consistency/runtime trade-off against our PINN baseline and a similarly sized MLP?** Improvement is a hypothesis, not a promised outcome.

## 2. What the existing work provides

Reviewed: `README.md`, `AGENTS.md`, `REPRODUCTION_NOTES.md`, `ENVIRONMENT.md`, both PINN explanation documents, the full-reproduction plan and historical audit, saved current-run metrics, and the implemented current loss expressions. The [PINN4SOH paper](https://www.nature.com/articles/s41467-024-48779-z) supplies the published context for charging-feature SOH estimation; the architecture and experiments below are our proposed development choices.

| Available resource | Decision for development |
| --- | --- |
| Processed MIT data and working loader | Use these first; no raw-data extraction or extra dataset acquisition. |
| 16 charging features plus cycle index | Keep all 17 inputs and their existing meaning/order. |
| Capacity label divided by 1.1 Ah for MIT | Keep the same SOH fraction target. |
| Adjacent retained-row training pairs | Reuse for the new trend penalty. A retained pair may span removed cycles. |
| Existing CPU environment | Reuse it unchanged; no historical dependency reconstruction. |
| Current-code PINN Run 01 | Main fixed reference. |
| Historical run and audits | Background evidence, not an ongoing optimization target. |

Current-code reference, from `results/reproduction/MIT/run_01/metrics.json`:

| MAE | MAPE (%) | RMSE | MSE | Best / stopping epoch | Total runtime |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.0065642339 | 0.6927548908 | 0.0090979158 | 0.0000827721 | 65 / 86 | 218.76 s |

These are sample-weighted metrics on 14,170 samples, with SOH expressed as a fraction and MAPE using the ground-truth denominator. This is one unseeded reference run, not an estimate of PINN's mean performance. Do not compare a new multi-seed mean with this number as if both had equal statistical support.

## 3. Freeze the initial data protocol

Use the existing MIT loader and exactly the saved baseline battery ordering and membership: 102 batteries supply training/validation; 23 are held out for testing. Battery IDs divisible by five form the test set. Remaining adjacent pairs use the existing 80/20 split with `random_state=420`: 54,053 training and 13,514 validation pairs.

Keep cleaning, nominal-capacity conversion, normalization, feature order, labels and test sample selection unchanged. Validate the ordered file manifest and resulting tensor hashes against the baseline before training. Fail visibly on a mismatch rather than silently sorting or changing the split. Preserve battery/row identity as side metadata for plots, never as extra model inputs.

All candidate models see the same 17-dimensional vector `z`. It contains eight voltage-related features, eight current-related features, and normalized cycle index at column 16. There are **no supplied voltage-relaxation traces, separate temperature input, or electrochemical mechanism labels**. Do not invent them or describe the charging features as relaxation features.

The task remains estimating SOH for a cycle whose charging features are available. It is not remaining useful life prediction or a forecast of future SOH without future features.

Known inherited limits must accompany results: validation shares battery identities and overlapping adjacent samples with training; normalization uses each battery's full trajectory, including test trajectories; cleaning consults capacity labels. The first comparison is therefore an **offline comparison under the inherited protocol**, not proof of deployable online prognosis. A future battery-disjoint/causal protocol would be a separately named experiment, with all models evaluated under that new protocol. It is outside this first implementation.

## 4. First architecture: two experts and one small router

Start with two experts. More experts, sparse top-k routing, transformers and per-expert dynamics networks add unnecessary variables at this stage.

| Component | Proposed dimensions and behavior | Reason |
| --- | --- | --- |
| Expert 1 and expert 2 | Each `17 → 32 → 16 → 1`, tanh hidden activations, linear output | Small CPU-friendly SOH regressors; independent initialization lets them differ. |
| Router | `7 → 8 → 2`, tanh hidden activation, softmax output | Produces two nonnegative weights adding to one. |
| Mixture output | Weighted sum of the two SOH predictions | Continuous routing; both experts learn on each batch. |
| Dropout | None in the first new-model comparison | Reduces stochastic complications in paired trend calculations; explicitly differs from the preserved PINN. |
| Output bounds | No sigmoid or clipping | Existing SOH labels can exceed 1; avoid imposing an incorrect upper bound. |

Use the following **existing normalized columns** as router inputs:

| Index | Name | Intended information |
| ---: | --- | --- |
| 4 | CC Q | Charge passed in the selected constant-current charging window |
| 5 | CC charge time | Duration of that window |
| 6 | voltage slope | Voltage rise through the window |
| 12 | CV Q | Charge passed in the selected constant-voltage window |
| 13 | CV charge time | Duration of that window |
| 14 | current slope | Current decay through the window |
| 16 | cycle index | Position within the normalized recorded cycling trajectory |

These are interpretable aging-related proxies, not direct measurements of resistance, lithium loss or SEI growth. Per-battery normalization also limits absolute cross-battery physical interpretation. Choosing this subset is an explicit new router design; the experts still receive all 17 original features. No capacity label, future target, battery ID or test error enters the router.

For input `z`, let `e1(z)` and `e2(z)` be expert predictions, and `g1(z)` and `g2(z)` be router weights:

```text
predicted SOH = g1(z) × e1(z) + g2(z) × e2(z)
g1(z) + g2(z) = 1
```

For example, predictions 0.92 and 0.88 with weights 0.75 and 0.25 produce 0.91 SOH. Experts are not preassigned names such as “SEI expert” or “late-life expert.” Any specialization must be demonstrated after training, and remains associational.

```text
Same 17 features ──┬── Expert 1 ── SOH estimate 1 ──┐
                  └── Expert 2 ── SOH estimate 2 ──┤
Seven selected features ── Router ── two weights ──┤
                                                 ↓
                                        Weighted SOH estimate
                                                 ↓
                                  Fit labels + soft trend penalty
```

With biases, this design has 2,324 trainable parameters: 1,121 per expert plus 82 in the router. Verify the count in implementation. A plain `17 → 48 → 28 → 1` tanh MLP has 2,265 parameters, within about 3%, and serves as the main capacity-matched control.

## 5. A simple, explicit physics contribution

The first version uses **data fit plus a soft non-increasing-trend prior**. It omits the PINN's learned dynamics network and PDE residual. This is an intentional new-model simplification, not a correction to the baseline or an equivalent implementation of its equations.

For a batch of `B` chronologically ordered same-battery pairs, use mixture predictions `u1`, `u2` and true SOH values `y1`, `y2`:

```text
L_data  = 0.5 × mean((u1-y1)^2) + 0.5 × mean((u2-y2)^2)
L_trend = mean(ReLU(u2-u1)^2)
L_total = L_data + lambda_trend × L_trend
```

`mean` averages over the batch; `ReLU(a)` is the positive part of `a`. The trend loss penalizes predicted rises and allows predicted declines. Apply it to the **final mixture**, so both experts and router receive its gradients. A weighted mixture of individually monotone experts is not automatically monotone when routing weights change.

Start with `lambda_trend=1` as a declared pilot choice, not an optimum. Both losses use mean squared SOH-fraction quantities, making their scaling inspectable. Do not reuse historical beta 50: that coefficient multiplied a different, summed, label-only expression. The new term depends on predictions and must pass a nonzero-gradient check on deliberately increasing predictions.

Battery degradation motivates a broad downward trend, but observed capacity can recover temporarily and measurements are noisy. This is therefore a **soft empirical prior**, not an exact electrochemical law. It does not enforce monotonicity, constrain the decline rate, or guarantee physical validity. Its usefulness must be tested against data-only controls. It also penalizes changes over retained pairs, not a derivative in physical units; pair gaps must be disclosed.

Initially add no PDE term, expert-separation penalty or router-balancing loss. Record expert utilization instead. If the router collapses onto one expert, first report it and inspect whether that expert already performs adequately. Any later balancing penalty is a separate declared experiment, not an invisible rescue mechanism.

Physics-related features alone do not demonstrate a physically constrained model. The positive-weight trend term supplies the explicit physical prior here. The router and expert functions remain learned from data. Use the report name **“simplified physics-regularized MoE (PIMOE prototype)”** and describe these limits.

## 6. Small experiment matrix

Retain the existing PINN result without retraining. Compare four new models under identical new-model training settings:

| ID | Architecture | Trend weight | Question |
| --- | --- | ---: | --- |
| M0 | Capacity-matched MLP | 0 | How well does a small single network perform? |
| M1 | Same MLP | 1 | Does the trend prior help without experts? |
| E0 | Two-expert MoE | 0 | Does learned routing help without the prior? |
| E1 | Same MoE | 1 | Does combining routing and the prior help? |

E1 versus E0 isolates the trend loss within the MoE. E0 versus M0 tests the mixture design at similar parameter counts. E1 versus M1 tests whether experts add value when both models receive the prior. Differences against PINN include architecture, optimizer schedule and loss design, so are an overall engineering comparison, not a single-factor ablation.

Do not add a router-input sweep or extra expert counts initially. A later uniform-average control would help separate learned routing from ensembling if the MoE shows a useful gain. Defer that until the initial matrix justifies it.

## 7. Proposed training protocol and development gates

These settings apply only to future new-model runs:

- Current CPU environment, float32, batch size 512, `num_workers=0`.
- One Adam optimizer over all model parameters: LR 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0; constant LR initially, no scheduler.
- Maximum 200 epochs; strict validation-MSE improvement; stop after 21 consecutive non-improving epochs to match the baseline's patience interpretation. Select a deep-copied/immediately serialized best checkpoint.
- Explicit training seeds 42, 43 and 44 for the final development matrix. Seed Python, NumPy and torch; seed CUDA if a later declared device change occurs. Use an explicitly seeded DataLoader generator separate from initialization RNG and record states. Keep split seed 420 unchanged.
- Validation/test use the same first-member sample selection as the baseline. Use validation only for choices. Evaluate the held-out test once per final selected run after restoring its best checkpoint.

| Stage | Bounded work after implementation is requested | Exit condition |
| --- | --- | --- |
| A: implementation and checks | Isolated model/trainer, data-manifest verification, small forward/gradient checks | Correct shapes, finite outputs, normalized gates, live expert/router gradients, correct checkpoint restore |
| B: smoke tests | Three epochs per matrix model on one existing split, seed 42 | Training/validation run; trend penalty behaves correctly; losses and gates logged. Smoke metrics are not reproduction claims. |
| C: validation pilot | At most 20 epochs per matrix model, seed 42 | Inspect stability, data/trend loss scale, runtime and collapse; no test-based selection |
| D: frozen comparison | Four variants × three seeds = at most 12 new-model runs, each capped at 200 epochs | Save every run and compare aggregate accuracy, trend behavior and CPU cost |
| E: FYP write-up | Tables, trajectory plots, router diagnostics and limitations | Explain benefits or negative results without overstating mechanism identification |

These are gates for future work, not permission to launch stages automatically now. If the pilot shows the trend term overwhelms fit or clearly worsens validation, allow one declared alternative `lambda_trend=0.1` for both M1/E1 under the same pilot budget. Record both attempts and freeze the choice before final testing. No open-ended search, seed selection or archive matching.

Use the three-epoch timing to estimate cost before stage D. The baseline took about 3.65 minutes, but that is not a reliable prediction for a new trainer. Ordinary MoE backpropagation avoids the PINN input-derivative graph, so lower per-epoch CPU cost is a reasonable hypothesis; measure it. Worst-case epoch budget is 12 × 200 = 2,400, with early stopping potentially reducing it.

## 8. What to measure and save

For each final run, report MAE, MAPE (%), RMSE and MSE using the standard ground-truth convention. Report sample-weighted results comparable to baseline plus clearly separated per-battery metrics and equal-battery summaries. Across seeds report individual results and mean ± sample standard deviation; do not treat correlated cycles as independent statistical replicates.

Also save:

- Training data/trend/weighted-total loss and validation MSE by epoch; best and stopping epoch.
- Configuration, dependency versions, source revision/diff, seeds/RNG states, ordered file/split manifests and tensor hashes.
- Best checkpoint and verified reload; predictions, true labels, battery/row IDs; parameter count, training runtime and CPU inference timing with a stated timing procedure.
- True/predicted SOH trajectories and residual plots per battery, sorted using retained row identity without changing training data.
- Mean gate weights, gate entropy, expert predictions and weights versus cycle coordinate. Report usage over many samples; one confident sample is not evidence of collapse.
- Predicted positive-step frequency and magnitude on adjacent evaluated samples, with the observed-label positive-step frequency alongside them. A lower violation rate alone can result from a nearly constant bad predictor, so always interpret it with accuracy.

Because the test results have already been seen for the baseline, avoid repeated development decisions using that benchmark. Gate visualizations from test data belong to final analysis, not iterative model selection.

## 9. Success criteria and limits

A useful FYP result need not beat the paper. Successful development means a functioning, traceable model, a controlled test of mixture and physics contributions, and an honest engineering explanation.

- Prefer a candidate whose validation performance is competitive and stable across seeds, with a useful trend-consistency or runtime benefit.
- Report whether final test errors improve on the saved PINN reference, but describe that comparison as preliminary because the reference is one unseeded run.
- If MoE does not improve over the matched MLP, retain that negative result. Do not add experts simply to obtain a favorable number.
- If the trend loss harms fit around real recovery, document the trade-off. It may establish that this simple prior is unsuitable at the adjacent-cycle scale.
- Do not interpret router weights as uncertainty estimates, probabilities of physical mechanisms, or proof of distinct aging regimes.

Only after the first matrix is understood should the project consider a second dataset, a causal evaluation protocol, relaxed trend penalties, or a shared dynamics network. Each introduces a new scientific question. None is required for the first PIMOE prototype.

## 10. Isolation and next deliverable

Suggested future layout, **not created in this planning task**:

```text
models/pimoe.py                    # experts and router
models/pimoe_controls.py           # matched MLP
scripts/train_pimoe.py             # explicit model/run selection; no hidden run loop
configs/pimoe/                    # declared pilot/final settings
tests/test_pimoe.py               # meaningful gradient/gate/checkpoint checks
results/smoke_tests/pimoe/         # bounded execution tests
results/pimoe/<variant>/<seed>/   # separate immutable development runs
```

No modifications to `Model/Model.py`, original main scripts, historical implementation, existing results or existing documentation are proposed. Record all future new-model departures in separate development documentation rather than rewriting reproduction history. Use fresh output directories and fail rather than overwrite.

**Recommended next deliverable:** after this plan is accepted, implement only the isolated two-expert model, matched MLP, trainer and checks, then perform the bounded three-epoch smoke stage. Review that evidence before progressing to the pilot and full comparison. Historical-environment work stays closed.
