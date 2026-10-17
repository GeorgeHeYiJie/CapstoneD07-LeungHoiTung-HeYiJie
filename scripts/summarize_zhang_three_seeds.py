"""Analyze six completed Zhang runs without loading models or training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/high_dimension_comparison'
DEST=OUT/'zhang_three_seed'
SEEDS=[42,7,123]
ARMS=['Z-O','Z-H']
METRICS=['MAE','MAPE_percent','RMSE','MSE']

def table(frame):
    rows=['| '+' | '.join(frame.columns)+' |','|'+'|'.join([' --- ']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.7g}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(rows)

def main():
    inputs=[OUT/a/f'seed_{s}'/f for s in SEEDS for a in ARMS for f in ['metrics.json','test_predictions.npz','loss_history.csv']]
    assert all(p.exists() for p in inputs),'All six completed runs required'
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    DEST.mkdir(exist_ok=True) # Refresh derived analysis only; raw run artifacts are read-only.
    all_runs=[];battery_rows=[];paired=[];diagnostics=[];profiles=[];samples={}
    lengths={p.stem:len(pd.read_csv(p)) for p in (ROOT/'data/MIT data/2018-04-12').glob('*.csv')}
    reference=dict(np.load(OUT/'Z-O/seed_42/test_predictions.npz'))
    for seed in SEEDS:
        preds={}
        for arm in ARMS:
            folder=OUT/arm/f'seed_{seed}'
            m=json.loads((folder/'metrics.json').read_text())
            assert m['seed']==seed and m['stopping_epoch']==200 and m['checkpoint_verified']
            all_runs.append(m)
            p=dict(np.load(folder/'test_predictions.npz'))
            for k in ['y_true','battery_id','row_index']:assert np.array_equal(p[k],reference[k]),(seed,arm,k)
            y=p['y_true'].ravel().astype(float);pred=p['y_pred'].ravel().astype(float)
            err=pred-y
            calculated=dict(MAE=np.abs(err).mean(),MAPE_percent=100*np.abs(err/y).mean(),RMSE=np.sqrt((err**2).mean()),MSE=(err**2).mean())
            for k,v in calculated.items():assert np.isclose(v,m[k],rtol=1e-12,atol=1e-14)
            d=pd.DataFrame(dict(battery=p['battery_id'],cycle=p['row_index'],y=y,pred=pred,error=err,absolute=np.abs(err),squared=err**2))
            assert not d.duplicated(['battery','cycle']).any()
            d['progress']=d.cycle/(d.battery.map(lengths)-1)
            preds[arm]=d;samples[seed,arm]=d
            for battery,g in d.groupby('battery'):
                early=g[g.progress<.2]
                battery_rows.append(dict(seed=seed,arm=arm,battery=battery,n=len(g),MAE=g.absolute.mean(),MAPE_percent=100*(g.absolute/g.y).mean(),RMSE=np.sqrt(g.squared.mean()),MSE=g.squared.mean(),early_MAE=early.absolute.mean(),early_bias=early.error.mean()))
            for interval,g in d.groupby(pd.cut(d.progress,np.linspace(0,1.000001,6),right=False),observed=True):
                profiles.append(dict(seed=seed,arm=arm,bin=str(interval),position=g.progress.mean(),MAE=g.absolute.mean(),bias=g.error.mean(),n=len(g)))
        per=pd.DataFrame([r for r in battery_rows if r['seed']==seed])
        o=per[per.arm=='Z-O'].set_index('battery');h=per[per.arm=='Z-H'].set_index('battery')
        delta=h.MAE-o.MAE;excess=delta*o.n
        positive=excess.clip(lower=0).sum()
        b39='2018-04-12_battery-39'
        early_o=preds['Z-O'].query('progress < .2');early_h=preds['Z-H'].query('progress < .2')
        diagnostics.append(dict(seed=seed,batteries_improved=int((delta<0).sum()),batteries_worsened=int((delta>0).sum()),batteries_tied=int((delta==0).sum()),early_batteries_worsened=int((h.early_MAE>o.early_MAE).sum()),early_original_MAE=early_o.absolute.mean(),early_matrix_MAE=early_h.absolute.mean(),early_matrix_bias=early_h.error.mean(),battery39_original_MAE=o.loc[b39,'MAE'],battery39_matrix_MAE=h.loc[b39,'MAE'],battery39_improvement_percent=100*(o.loc[b39,'MAE']-h.loc[b39,'MAE'])/o.loc[b39,'MAE'],battery39_positive_excess_share_percent=100*max(0,excess[b39])/positive if positive else 0,battery39_matrix_squared_error_share_percent=100*(h.loc[b39,'MSE']*h.loc[b39,'n'])/(h.MSE*h.n).sum(),battery39_excess_rank=int(excess.rank(ascending=False,method='min')[b39]) if excess[b39]>0 else None,largest_excess_battery=excess.idxmax() if (excess>0).any() else 'none (all batteries improve)'))
        for b in o.index:
            paired.append(dict(seed=seed,battery=b,MAE_original=o.loc[b,'MAE'],MAE_matrix=h.loc[b,'MAE'],MAE_difference=delta[b],MAE_improvement_percent=-100*delta[b]/o.loc[b,'MAE'],excess_absolute_error=excess[b]))
    runs=pd.DataFrame(all_runs);per=pd.DataFrame(battery_rows);diag=pd.DataFrame(diagnostics);prof=pd.DataFrame(profiles)
    runs.to_csv(DEST/'metrics_by_run.csv',index=False)
    per.to_csv(DEST/'per_battery_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(DEST/'paired_battery_improvements.csv',index=False)
    diag.to_csv(DEST/'seed_diagnostics.csv',index=False)
    prof.to_csv(DEST/'early_cycle_profiles.csv',index=False)
    aggregate=[];imp=[]
    for arm,g in runs.groupby('arm',sort=False):
        r=dict(model='Zhang',arm=arm,seeds='42;7;123',n_seeds=3,aggregation='mean of per-run pooled metrics; sample SD ddof=1')
        for k in METRICS+['training_runtime_seconds','total_arm_seconds','best_epoch']:
            r[k+'_mean']=g[k].mean();r[k+'_std']=g[k].std(ddof=1)
        aggregate.append(r)
    for k in METRICS:
        o=runs[runs.arm=='Z-O'].set_index('seed')[k];h=runs[runs.arm=='Z-H'].set_index('seed')[k]
        vals=100*(o-h)/o
        imp.append(dict(metric=k,original_mean=o.mean(),original_std=o.std(ddof=1),matrix_mean=h.mean(),matrix_std=h.std(ddof=1),paired_improvement_mean_percent=vals.mean(),paired_improvement_std_percent=vals.std(ddof=1),ratio_of_mean_errors_improvement_percent=100*(o.mean()-h.mean())/o.mean(),**{f'seed_{s}_improvement_percent':vals[s] for s in SEEDS}))
    agg=pd.DataFrame(aggregate);imps=pd.DataFrame(imp)
    previous=pd.read_csv(OUT/'zhang_shared_seed_7/previous_final_comparison.csv')
    wang=previous[previous.model=='Wang'].copy()
    assert len(wang)==2
    agg['model']='Zhang-inspired SOH adaptation'
    agg['matrix_vs_original_improvement_percent_definition']='ratio of mean errors'
    for k in METRICS:
        agg[k+'_improvement_percent']=np.nan
        agg.loc[agg.arm=='Z-H',k+'_improvement_percent']=100*(agg.loc[agg.arm=='Z-O',k+'_mean'].iloc[0]-agg.loc[agg.arm=='Z-H',k+'_mean'].iloc[0])/agg.loc[agg.arm=='Z-O',k+'_mean'].iloc[0]
    pd.concat([wang,agg],ignore_index=True).to_csv(OUT/'final_comparison.csv',index=False)
    imps.to_csv(DEST/'metric_improvements.csv',index=False)
    means=per.groupby(['arm','battery']).MAE.mean().unstack('arm')
    means['MAE_difference']=means['Z-H']-means['Z-O']
    means.to_csv(DEST/'per_battery_three_seed_mean.csv')
    fig,axs=plt.subplots(1,4,figsize=(13,4))
    for ax,k in zip(axs,METRICS):
        for i,arm in enumerate(ARMS):
            values=runs[runs.arm==arm][k]
            ax.bar(i,values.mean(),yerr=values.std(ddof=1),capsize=5,alpha=.65)
            ax.scatter(np.repeat(i,3),values,c='black',s=20)
        ax.set_xticks([0,1],['Original','Matrix']);ax.set_title(k);ax.set_ylim(bottom=0)
    fig.suptitle('Zhang: mean +/- sample SD across seeds 42, 7, 123; dots are individual runs')
    fig.tight_layout();fig.savefig(DEST/'three_seed_metrics.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for j,seed in enumerate(SEEDS):
        for arm in ARMS:
            d=samples[seed,arm];b=d[d.battery=='2018-04-12_battery-39']
            axs[0,j].plot(b.cycle,b.pred,label=arm,alpha=.8)
            q=prof[(prof.seed==seed)&(prof.arm==arm)]
            axs[1,j].plot(q.position,q.MAE,marker='o',label=arm)
        axs[0,j].plot(b.cycle,b.y,c='black',label='True')
        axs[0,j].set(title=f'Battery 39, seed {seed}',xlabel='CSV row index',ylabel='SOH fraction')
        axs[1,j].set(title=f'Error vs trajectory position, seed {seed}',xlabel='Relative recorded position',ylabel='MAE')
        axs[0,j].legend();axs[1,j].legend()
    fig.tight_layout();fig.savefig(DEST/'battery39_and_early_cycles.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    stability=[]
    for ax,seed in zip(axs,SEEDS):
        for arm in ARMS:
            q=pd.read_csv(OUT/arm/f'seed_{seed}/loss_history.csv')
            assert len(q)==200 and q.epoch.tolist()==list(range(1,201))
            jumps=q.loc[(q.epoch>30)&(q.validation_MSE/q.validation_MSE.shift(1)>5),'epoch'].tolist()
            best=q.loc[q.validation_MSE.idxmin()]
            stability.append(dict(seed=seed,arm=arm,best_epoch=int(best.epoch),final_to_best_validation_ratio=q.validation_MSE.iloc[-1]/best.validation_MSE,postwarmup_validation_jumps_over_5x=';'.join(map(str,jumps))))
            ax.semilogy(q.epoch,q.validation_MSE,label=arm)
            ax.scatter([best.epoch],[best.validation_MSE],marker='x')
        ax.set(title=f'Seed {seed}',xlabel='Epoch',ylabel='Validation MSE');ax.legend()
    fig.tight_layout();fig.savefig(DEST/'validation_by_seed.png',dpi=160);plt.close(fig)
    stability=pd.DataFrame(stability)
    stability.to_csv(DEST/'training_stability.csv',index=False)
    display=imps[['metric']].copy()
    for name in ['original','matrix']:display[name+' mean +/- SD']=[f'{r[name+"_mean"]:.7g} +/- {r[name+"_std"]:.4g}' for r in imp]
    display['paired improvement % mean +/- SD']=[f'{r["paired_improvement_mean_percent"]:+.3f} +/- {r["paired_improvement_std_percent"]:.3f}' for r in imp]
    tail=[]
    for seed in SEEDS:
        for arm in ARMS:
            g=samples[seed,arm];b=g[g.battery=='2018-04-12_battery-39']
            worst=b.loc[b.absolute.idxmax()]
            tail.append(dict(seed=seed,arm=arm,battery39_MAE=b.absolute.mean(),battery39_squared_error_share_percent=100*b.squared.sum()/g.squared.sum(),max_absolute_error=b.absolute.max(),worst_row=int(worst.cycle),true_at_worst=worst.y,prediction_at_worst=worst.pred))
    tail=pd.DataFrame(tail);tail.to_csv(DEST/'battery39_tail_errors.csv',index=False)
    for seed in SEEDS:
        shared=OUT/f'zhang_shared_seed_{seed}'
        for name in ['split_manifest.json','scaler.json','dataset_manifest.json']:
            assert json.loads((shared/name).read_text(encoding='utf-8'))==json.loads((OUT/'zhang_shared_seed_42'/name).read_text(encoding='utf-8'))
        for name in ['valid_references.npz','test_references.npz']:
            old=np.load(OUT/'zhang_shared_seed_42'/name);new=np.load(shared/name)
            assert all(np.array_equal(old[k],new[k]) for k in old.files)
    wang_display=pd.DataFrame({'metric':METRICS})
    for arm in ['W-O','W-H']:
        r=wang[wang.arm==arm].iloc[0]
        wang_display[arm+' mean +/- SD']=[f'{r[k+"_mean"]:.7g} +/- {r[k+"_std"]:.4g}' for k in METRICS]
    lines=['# MIT comparison: Wang and Zhang three-seed results','',
        'Latest stage: Zhang robustness only, seeds 7 and 123 added to existing seed 42. Four new runs completed. No Wang rerun, architecture changes, tuning or ablation. Each arm completed 200 epochs; checkpoint selection used its own lowest validation mixed-prediction MSE.','',
        '## Frozen protocol and verification','',
        'HD-SOH-v1 is unchanged: 71/18/36 MIT train/validation/test batteries; 42,784/13,585/29,083 pairs; split seed 420; SOH=capacity/1.1; train-only min-max; causal feature filling and 16-cycle windows. Test battery/row keys and labels are identical in all six Zhang runs and match the existing Wang protocol. All metrics were independently recalculated from saved predictions.','',
        'Zhang is the existing BatLiNet-inspired SOH adaptation, not the original paper\'s lifetime model. Original and matrix use the same existing classes as seed 42 (9,313 versus 18,529 parameters). Adam lr=0.001, batch 512, no scheduler or early stopping; losses, mixture weight 0.5 and two evaluation references are unchanged. CPU thread counts remain 8.','',
        'Training initialization and batch order follow seeds 42/7/123; training-reference RNG is 10000+seed. Within each seed, original/matrix receive the same batch-order and training-reference schedule. Validation and test references remain frozen across ALL seeds (RNG 20042 and 30042), not merely within each paired seed. Saved reference IDs and clamped rows were verified identical. Only training batteries provide reference labels. This isolates training randomness from evaluation-reference randomness.','',
        'MAE/RMSE use SOH fractions; MSE uses squared fractions; MAPE is percent. Mean +/- SD is computed over three per-run pooled metrics, with sample SD (ddof=1). Positive improvement means matrix is better. Paired improvement is averaged after calculating each seed separately; ratio-of-mean-error improvement is separately saved in final_comparison.csv and metric_improvements.csv. These are different statistics.','',
        '## Zhang three-seed metrics','',table(display),'',
        '![Three-seed metrics](zhang_three_seed/three_seed_metrics.png)','',
        '## Individual runs','',table(runs[['arm','seed']+METRICS+['training_runtime_seconds','total_arm_seconds','best_epoch','stopping_epoch']]),'',
        f"The four additional runs used {runs[runs.seed!=42].training_runtime_seconds.sum()/60:.2f} training minutes and {runs[runs.seed!=42].total_arm_seconds.sum()/60:.2f} total-arm minutes (including validation, evaluation and saving). Setup/analysis are outside those totals. All checkpoints passed exact validation-prediction reload checks.",'',
        '## Per-battery consistency','',table(diag[['seed','batteries_improved','batteries_worsened','early_batteries_worsened','largest_excess_battery']]),'',
        f"Comparing each battery's three-seed mean MAE, {(means.MAE_difference<0).sum()}/36 improve and {(means.MAE_difference>0).sum()}/36 worsen. All official metrics include all batteries; no outcome-based exclusions were applied.",'',
        '## Battery 39 across seeds','',table(diag[['seed','battery39_original_MAE','battery39_matrix_MAE','battery39_improvement_percent','battery39_positive_excess_share_percent','battery39_matrix_squared_error_share_percent','battery39_excess_rank']]),'',
        table(tail),'',
        'Excess-absolute-error share uses the sum of positive excess across worsened batteries as denominator; squared-error share uses the entire matrix test set. Averages can hide a few extreme samples, so the maximum error and exact row/true/predicted SOH are reported separately. Battery 39\'s late input distribution shift was documented in the seed-42 analysis; this analysis does not establish a causal mechanism.','',
        '## Early-cycle robustness','',table(diag[['seed','early_original_MAE','early_matrix_MAE','early_matrix_bias','early_batteries_worsened']]),'',
        'Early means the first 20% of each recorded CSV trajectory. Recorded length is used only for retrospective grouping, never as a model input or normalization scale. Positive signed bias means overprediction. Compare both pooled early error and the number of affected batteries rather than assuming uniform behavior.','',
        '![Battery 39 and trajectory-position errors](zhang_three_seed/battery39_and_early_cycles.png)','',
        '## Training stability','',table(stability),'',
        'The jump rule is descriptive: validation MSE >5 times the previous epoch after epoch 30. Persistent failure requires inspecting recovery and final/best ratios, not just a spike flag.','',
        '![Validation by seed](zhang_three_seed/validation_by_seed.png)','',
        '## Wang reference results (unchanged; no reruns)','',table(wang_display),'',
        'Wang per-run and per-battery results remain in wang_three_seed/. The complete previous report, including Wang interpretation and the Zhang seed-42 analysis, is preserved in zhang_shared_seed_7/previous_analysis_summary.md. Existing predictions, checkpoints and individual-stage reports are not overwritten.','',
        *['## Interpretation of the completed robustness check', '', "The matrix advantage is seed-dependent. MAE improvements for seeds 42, 7 and 123 are +19.15%, -10.28% and +20.32%; all four pooled metrics favor matrix in seeds 42 and 123 but favor original in seed 7. Battery improvement counts are 34/36, 8/36 and 36/36. Comparing each battery's mean MAE across seeds gives 31/36 improved, but this does not make the gain consistent in every run.", '', 'The ratio of mean errors gives +11.45% MAE, +11.84% MAPE, +3.08% RMSE and +6.98% MSE improvement. However, mean paired percentage improvements are +9.73%, +9.95%, +0.15% and -4.88% respectively, with large sample SDs. The MSE sign difference is legitimate: averaging ratios weights each seed relative to its own original error, whereas taking the ratio of mean errors is a different aggregation. Seed 7 has a low original MSE and a large relative deterioration. Both definitions are reported to avoid presenting the aggregate as uniform success.', '', 'Battery 39 remains a serious but seed-dependent robustness concern. Matrix MAE is 0.04477, 0.05322 and 0.01849 for seeds 42/7/123, compared with original 0.03376, 0.02259 and 0.03372. Matrix is worse in the first two seeds and improves by 45.16% in seed 123. Its maximum sample error is 0.97872, 1.61126 and 0.17648 SOH units respectively. At row 90 (true SOH 0.802718), matrix predicts 1.781437, 2.413981 and 0.979196. Battery 39 contributes 21.05%, 34.71% and 0.84% of matrix squared error. The severe tail is reproducible in two seeds but is not inevitable in every initialization.', '', 'Seed 7 is not explained solely by Battery 39: 28/36 batteries worsen in full-trajectory MAE, and Battery 33 is the largest contributor to excess absolute error. Battery 39 ranks ninth by additive excess absolute error but dominates matrix squared error because a few very large errors are heavily penalized. Seed 123 has no positive excess-error contributor because every battery improves; its excess rank is not applicable.', '', 'Early-cycle weakness is not universal across seeds. Pooled early MAE changes by +7.37%, -1.21% and +2.04% (positive means worse), with 28, 19 and 21 of 36 batteries worsening early. All six early signed biases are positive: both representations tend to overpredict early SOH. Thus a broad early bias persists, but matrix is not worse in pooled early MAE in every seed; seed 7 has a small pooled improvement despite 19 batteries worsening.', '', 'No Zhang seed exhibits the persistent training collapse observed in Wang matrix. Final/best validation MSE ratios are 1.01-1.17 for matrix and 1.11-1.19 for original. Seed-7 matrix has a >5x post-warmup jump at epoch 134, but recovers and reaches its best at epoch 199. Stability of training therefore does not guarantee robust held-out predictions: the Battery-39 failures still occur at validation-selected checkpoints.', '', 'Conclusion: the three-seed averages support a modest MAE/MAPE benefit, but the current matrix pipeline is not a reliably superior replacement. RMSE/MSE gains are small and inconsistent, and catastrophic local errors remain sensitive to training randomness. Three seeds on one fixed test batch are exploratory evidence, not a significance claim. No architecture changes, clipping, data exclusions, tuning or ablation were performed. All requested training is complete and stopped.', ''],
        '## Limits and artifacts','',
        'Three training seeds on one fixed held-out MIT date batch do not establish population significance. Zhang original/matrix have different input channels/parameter counts; this is a pipeline comparison, not an FFT-only causal test. References require labeled historical training batteries. No tuning to test batteries was performed.','',
        'final_comparison.csv contains four three-seed aggregate rows: Wang original/matrix and Zhang original/matrix. zhang_three_seed/ contains all six per-run metrics, 216 per-battery metric rows, paired improvements, seed diagnostics, battery-39 tails, position profiles, stability history and plots. New raw artifacts are in Z-O/seed_7, Z-H/seed_7, Z-O/seed_123 and Z-H/seed_123. New configuration/reference manifests and source snapshots are in zhang_shared_seed_7 and zhang_shared_seed_123. Stop here: no new Wang runs, additional Zhang seeds or ablation.','']
    (OUT/'ANALYSIS_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    (DEST/'input_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(display.to_string(index=False));print(diag.to_string(index=False));print(tail.to_string(index=False));print(stability.to_string(index=False))

if __name__=='__main__':main()
