# Final conclusion: high-dimensional matrix analysis

This conclusion uses only the completed MIT experiments: Wang original/matrix and Zhang-inspired original/matrix, each with seeds 42, 7 and 123. No new training or ablation was performed for this document.

The frozen protocol used the same 71/18/36 train/validation/test batteries, SOH = capacity/1.1, training-derived normalization and 29,083 test samples. Each run used 200 epochs and its validation-selected checkpoint. Results describe this fixed MIT split, not general battery populations.

## 1. What is the matrix for?

It reorganizes recent battery measurements into a structured input so a model can potentially learn temporal patterns more effectively. A causal 16-cycle window of 16 features undergoes FFT-based period selection, reshaping, padding and resizing into two feature maps. This is an input representation experiment, not a new physical law.

## 2. What additional information does it provide?

For Wang, the original input contains the current 16 features and cycle/time. The matrix version also exposes recent history: 2,048 matrix values are appended to the original 17 values. It therefore changes both historical context and input size.

For Zhang-inspired, the original input already contains the same 16-cycle history. The matrix reorganizes that history into a different spatial layout; it does not add measurements. In both cases, extra dimensions are derived from existing data, not independent new physical information. Selected frequencies are not verified physical degradation mechanisms.

## 3. Did it improve Wang?

**No, for the tested configuration.** All four pooled metrics worsened in every seed. However, performance did not worsen on every battery: 21, 22 and 19 of 36 batteries improved in seeds 42, 7 and 123. Battery 33 was the largest contributor to excess absolute error in all three seeds. A few severe failures therefore coexist with benefits on many batteries. Pooled early-cycle MAE also worsened in all three seeds.

## 4. Did it improve Zhang-inspired?

**It produced a modest average improvement, but not a consistent improvement across seeds.** All four metrics improved in seeds 42 and 123, but worsened in seed 7. Battery improvement counts were 34/36, 8/36 and 36/36; 31/36 improved when comparing each battery's three-seed mean MAE.

This is the repository's BatLiNet/Zhang-inspired SOH adaptation, not a direct evaluation of the original Zhang et al. (2025) lifetime model.

| Model | Metric | Original mean ± sample SD | Matrix mean ± sample SD | Improvement |
| --- | --- | --- | --- | --- |
| Wang | MAE | 0.006777 ± 0.001440 | 0.009958 ± 0.002960 | −46.94% |
| Wang | MAPE (%) | 0.7284 ± 0.1559 | 1.0676 ± 0.3287 | −46.56% |
| Wang | RMSE | 0.008804 ± 0.001437 | 0.017660 ± 0.010102 | −100.60% |
| Wang | MSE | 0.00007888 ± 0.00002650 | 0.00037992 ± 0.00041465 | −381.62% |
| Zhang-inspired | MAE | 0.016540 ± 0.002805 | 0.014647 ± 0.001136 | +11.45% |
| Zhang-inspired | MAPE (%) | 1.7745 ± 0.3147 | 1.5643 ± 0.1241 | +11.84% |
| Zhang-inspired | RMSE | 0.019571 ± 0.003178 | 0.018969 ± 0.002030 | +3.08% |
| Zhang-inspired | MSE | 0.00038977 ± 0.00012215 | 0.00036255 ± 0.00007927 | +6.98% |

Each run's metrics pool all test samples; the table averages those metrics across seeds. MAE/RMSE use fractional SOH; MSE uses squared fractional SOH. Improvement = (original mean error − matrix mean error) / original mean error × 100. Negative means worse. This is a ratio of means, not the mean of seed-wise percentage improvements: for example, Zhang's mean seed-wise MSE improvement is −4.88%, reflecting the large relative deterioration in seed 7.

## 5. Why might the effects differ?

Wang flattens the matrix into a much wider MLP input, whereas Zhang-inspired applies convolution to its spatial layout. These networks may use the representation differently. Wang also gains history that its original input lacks; Zhang already has that history. Input-layer parameter counts change in both comparisons, and the two model families have different objectives and training procedures.

These are plausible explanations, not established causes. The experiments compare complete pipelines and do not isolate history, FFT, interpolation, model capacity or optimization effects.

## 6. Main weaknesses

- **Outlier and seed sensitivity:** Wang Battery 33 remained problematic. Zhang Battery 39 had extreme local errors in seeds 42 and 7, but improved in seed 123. Seed 7 also worsened on 28 batteries, so its failure cannot be attributed solely to Battery 39.
- **Early-cycle limitations:** Early means the first 20% of each recorded trajectory. Wang matrix was worse in all seeds. Zhang matrix early MAE was 7.37% worse, 1.21% better and 2.04% worse; both Zhang representations tended to overpredict early SOH.
- **Training stability:** Wang matrix showed persistent validation deterioration in all three runs. Reported test results use checkpoints selected before those collapses, so later spikes do not establish the cause of test errors. Zhang had no comparable persistent collapse; its seed-7 spike recovered, despite poor held-out predictions.
- **Higher cost:** Mean training time increased from 6.32 to 12.47 minutes for Wang and 14.89 to 22.38 minutes for Zhang. Mean total run time, including validation/evaluation/saving, increased from 6.66 to 13.17 and 18.09 to 26.56 minutes respectively on the recorded CPU environment.

## 7. What can we safely conclude?

The matrix is a model-dependent representation option, not an automatic accuracy improvement. Under this protocol it reduced Wang's overall accuracy, while giving Zhang-inspired a modest, seed-dependent average benefit with unresolved local failures. Its benefit must be assessed alongside per-battery reliability, early-cycle error and computational cost.

## 8. What should we NOT conclude?

We cannot claim that FFT caused the gains, that more dimensions necessarily add useful information, or that matrix inputs are universally better or worse. Three seeds on one held-out MIT batch do not establish statistical significance, generalization to other datasets, or superiority over either published paper. Larger input layers mean this was not a capacity-matched representation-only experiment.

## 9. What should be investigated next?

1. Compare FFT and non-FFT representations with the same history, model capacity, normalization and training budget to separate possible explanations.
2. Investigate Battery 33/39 feature trajectories and early-window behavior; inspect input scales and training gradients to diagnose failures without removing inconvenient test samples.
3. Evaluate any proposed stability or normalization changes using training/validation data, then assess the frozen design on additional seeds and untouched batteries or datasets. Since this test set has now been examined closely, it should not become the tuning set.

These are recommendations only; no further experiments were run.

## Reusable presentation text

### Short PPT paragraph

Across three seeds on the same MIT split, matrix inputs increased Wang's mean MAE by 46.9% but reduced Zhang-inspired mean MAE by 11.5%. Zhang's benefit was seed-dependent, and battery outliers, early-cycle errors and higher runtime remained. The matrix is therefore a potentially useful representation for Zhang-inspired, not a generally superior input. These experiments do not isolate FFT's contribution.

### Short FYP report paragraph

Under a frozen MIT evaluation protocol, three-seed comparisons showed that the high-dimensional matrix increased Wang's mean MAE from 0.00678 to 0.00996, while reducing Zhang-inspired mean MAE from 0.01654 to 0.01465. The latter improvement was not consistent across seeds, and severe battery-level errors remained. Matrix inputs also increased runtime, with persistent training deterioration observed for Wang. These findings support model-dependent utility rather than universal superiority. Because historical context, input geometry and parameter counts were not all held constant, the observed differences cannot be attributed specifically to FFT.

### Three oral presentation points

- The matrix reorganizes recent battery history; more dimensions do not mean new physical measurements.
- Wang's average MAE worsened by 46.9%; Zhang-inspired improved by 11.5%, but failed to improve consistently across seeds.
- Next we need matched representation controls and robustness checks before attributing benefits to FFT or recommending general use.

## Evidence used

- [Aggregate metrics](final_comparison.csv) and [completed analysis summary](ANALYSIS_SUMMARY.md).
- [Wang battery/early-cycle diagnostics](wang_three_seed/seed_diagnostics.csv) and [training stability](wang_three_seed/training_stability.csv).
- [Zhang battery/early-cycle diagnostics](zhang_three_seed/seed_diagnostics.csv), [Battery 39 tail errors](zhang_three_seed/battery39_tail_errors.csv) and [training stability](zhang_three_seed/training_stability.csv).
- [Frozen comparison design and representation limitations](../../docs/HIGH_DIMENSION_ANALYSIS_PLAN.md).
