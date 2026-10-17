"""Analyze six completed Wang runs without loading models or training."""
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
DEST=OUT/'wang_three_seed'
SEEDS=[42,7,123]
ARMS=['W-O','W-H']
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
    DEST.mkdir(exist_ok=False)
    all_runs=[];battery_rows=[];paired=[];diagnostics=[];profiles=[];samples={}
    lengths={p.stem:len(pd.read_csv(p)) for p in (ROOT/'data/MIT data/2018-04-12').glob('*.csv')}
    reference=dict(np.load(OUT/'W-O/seed_42/test_predictions.npz'))
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
        o=per[per.arm=='W-O'].set_index('battery');h=per[per.arm=='W-H'].set_index('battery')
        delta=h.MAE-o.MAE;excess=delta*o.n
        positive=excess.clip(lower=0).sum()
        b33='2018-04-12_battery-33'
        early_o=preds['W-O'].query('progress < .2');early_h=preds['W-H'].query('progress < .2')
        diagnostics.append(dict(seed=seed,batteries_improved=int((delta<0).sum()),batteries_worsened=int((delta>0).sum()),batteries_tied=int((delta==0).sum()),early_batteries_worsened=int((h.early_MAE>o.early_MAE).sum()),early_original_MAE=early_o.absolute.mean(),early_matrix_MAE=early_h.absolute.mean(),early_matrix_bias=early_h.error.mean(),battery33_original_MAE=o.loc[b33,'MAE'],battery33_matrix_MAE=h.loc[b33,'MAE'],battery33_improvement_percent=100*(o.loc[b33,'MAE']-h.loc[b33,'MAE'])/o.loc[b33,'MAE'],battery33_positive_excess_share_percent=100*max(0,excess[b33])/positive if positive else 0,battery33_matrix_squared_error_share_percent=100*(h.loc[b33,'MSE']*h.loc[b33,'n'])/(h.MSE*h.n).sum(),battery33_excess_rank=int(excess.rank(ascending=False,method='min')[b33]),largest_excess_battery=excess.idxmax()))
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
        r=dict(model='Wang',arm=arm,seeds='42;7;123',n_seeds=3,aggregation='mean of per-run pooled metrics; sample SD ddof=1')
        for k in METRICS+['training_runtime_seconds','total_arm_seconds','best_epoch']:
            r[k+'_mean']=g[k].mean();r[k+'_std']=g[k].std(ddof=1)
        aggregate.append(r)
    for k in METRICS:
        o=runs[runs.arm=='W-O'].set_index('seed')[k];h=runs[runs.arm=='W-H'].set_index('seed')[k]
        vals=100*(o-h)/o
        imp.append(dict(metric=k,original_mean=o.mean(),original_std=o.std(ddof=1),matrix_mean=h.mean(),matrix_std=h.std(ddof=1),paired_improvement_mean_percent=vals.mean(),paired_improvement_std_percent=vals.std(ddof=1),ratio_of_mean_errors_improvement_percent=100*(o.mean()-h.mean())/o.mean(),**{f'seed_{s}_improvement_percent':vals[s] for s in SEEDS}))
    agg=pd.DataFrame(aggregate);imps=pd.DataFrame(imp)
    agg.to_csv(OUT/'final_comparison.csv',index=False)
    imps.to_csv(DEST/'metric_improvements.csv',index=False)
    means=per.groupby(['arm','battery']).MAE.mean().unstack('arm')
    means['MAE_difference']=means['W-H']-means['W-O']
    means.to_csv(DEST/'per_battery_three_seed_mean.csv')
    fig,axs=plt.subplots(1,4,figsize=(13,4))
    for ax,k in zip(axs,METRICS):
        for i,arm in enumerate(ARMS):
            values=runs[runs.arm==arm][k]
            ax.bar(i,values.mean(),yerr=values.std(ddof=1),capsize=5,alpha=.65)
            ax.scatter(np.repeat(i,3),values,c='black',s=20)
        ax.set_xticks([0,1],['Original','Matrix']);ax.set_title(k);ax.set_ylim(bottom=0)
    fig.suptitle('Wang: mean +/- sample SD across seeds 42, 7, 123; dots are individual runs')
    fig.tight_layout();fig.savefig(DEST/'three_seed_metrics.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for j,seed in enumerate(SEEDS):
        for arm in ARMS:
            d=samples[seed,arm];b=d[d.battery=='2018-04-12_battery-33']
            axs[0,j].plot(b.cycle,b.pred,label=arm,alpha=.8)
            q=prof[(prof.seed==seed)&(prof.arm==arm)]
            axs[1,j].plot(q.position,q.MAE,marker='o',label=arm)
        axs[0,j].plot(b.cycle,b.y,c='black',label='True')
        axs[0,j].set(title=f'Battery 33, seed {seed}',xlabel='CSV row index',ylabel='SOH fraction')
        axs[1,j].set(title=f'Error vs trajectory position, seed {seed}',xlabel='Relative recorded position',ylabel='MAE')
        axs[0,j].legend();axs[1,j].legend()
    fig.tight_layout();fig.savefig(DEST/'battery33_and_early_cycles.png',dpi=160);plt.close(fig)
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
    lines=['# Wang MIT comparison: three seeds','',
        'Completed seeds 42, 7 and 123 for W-O and W-H. No Zhang experiments and no FFT/no-FFT ablation were run. Existing seed-42 outputs and the detailed single-seed report remain unchanged.','',
        '## Protocol and verification','',
        'HD-SOH-v1: identical 71/18/36 train/validation/test batteries; 42,784/13,585/29,083 pairs; split seed 420; SOH=capacity/1.1; train-only min-max; batch 512; 200 epochs without early stopping; CPU with 8 threads. Each run uses its own lowest validation-MSE checkpoint. All six test key/label sequences match exactly, and saved metrics were independently recomputed from predictions.','',
        'Only training seeds changed. Python, NumPy, torch initialization/dropout and batch-order randomness follow each seed; the split is unchanged. The original helper and model code were not modified. Wang losses and both Adam optimizers/scheduler remain unchanged. The user-selected seeds 7 and 123 supersede the tentative 43/44 listed in the initial plan.','',
        'MAE/RMSE are SOH fractions; MSE is squared SOH fraction; MAPE is a percentage with true SOH as denominator. Reported SD is sample SD (ddof=1) across three independent training runs, not across cycles. Positive improvement means matrix is better. Paired improvement is calculated separately per seed then averaged; ratio-of-mean-errors is also saved separately.','',
        '## Three-seed metrics','',table(display),'',
        '![Three-seed comparison](wang_three_seed/three_seed_metrics.png)','',
        '## Individual runs','',table(runs[['arm','seed']+METRICS+['training_runtime_seconds','total_arm_seconds','best_epoch','stopping_epoch']]),'',
        'Runtime includes batch preparation/cache misses; validation is included in total-arm time and recorded separately in the CSV. It is measured, not inferred from the previous estimate.','',
        '## Battery-level consistency','',table(diag[['seed','batteries_improved','batteries_worsened','early_batteries_worsened','largest_excess_battery']]),'',
        f'Using each battery\'s mean MAE over the three seeds, {(means.MAE_difference<0).sum()}/36 improve and {(means.MAE_difference>0).sum()}/36 worsen. These are exploratory test-set diagnostics, not selection rules.','',
        '## Battery 33','',table(diag[['seed','battery33_original_MAE','battery33_matrix_MAE','battery33_improvement_percent','battery33_excess_rank','battery33_positive_excess_share_percent','battery33_matrix_squared_error_share_percent']]),'',
        'Excess absolute error is sample count times (matrix MAE - original MAE). Positive-excess share is relative to the sum over worsened batteries only; rank 1 is the largest excess contributor. All official metrics retain all 36 batteries.','',
        '## Early-cycle weakness','',table(diag[['seed','early_original_MAE','early_matrix_MAE','early_matrix_bias','early_batteries_worsened']]),'',
        'Early means the first 20% of each recorded CSV trajectory, matching the previous detailed analysis. The full recorded length is used only for retrospective grouping, never as model input. Positive signed bias means overprediction. This is recorded position, not a verified physical-age or end-of-life definition.','',
        '![Battery 33 and position profiles](wang_three_seed/battery33_and_early_cycles.png)','',
        '## Training stability','',table(stability),'',
        'The jump rule is descriptive: validation MSE greater than 5 times the previous epoch after epoch 30, matching the earlier audit. This does not identify a causal optimizer defect. Test metrics use the best validation epoch, not the final epoch.','',
        '![Validation by seed](wang_three_seed/validation_by_seed.png)','',
        *['## Interpretation of the completed runs', '', 'The matrix pipeline has worse pooled MAE, MAPE, RMSE and MSE in all three paired seeds. The ratio of mean errors gives improvements of -46.94% (MAE), -46.56% (MAPE), -100.60% (RMSE) and -381.62% (MSE). These ratios differ from the mean of per-seed percentage improvements reported above; both are retained and explicitly labeled.', '', "The result is not a uniform loss on every battery: 21, 22 and 19 of 36 batteries improve for seeds 42, 7 and 123 respectively, and 22 improve when comparing each battery's three-seed mean MAE. Nevertheless, severe errors recur in the same difficult battery. Battery 33 ranks first in excess absolute error in every seed, accounting for 75.45%, 36.54% and 60.99% of positive excess absolute error. Its matrix MAE is 0.02742, 0.01742 and 0.07124, versus original MAE 0.00247, 0.00376 and 0.00480. Seed 123 is particularly severe: this battery contributes 83.02% of matrix squared error. This is a reproducible failure concentration with seed-dependent severity, not merely a one-seed outlier.", '', 'Pooled early-cycle MAE is worse for matrix in all three seeds, with positive signed bias each time. However, the breadth varies: early MAE worsens on 34/36, 36/36 and 18/36 batteries. Therefore widespread early weakness reproduces strongly in seeds 42 and 7; seed 123 has a smaller pooled early penalty and only half the batteries worsen early. Do not describe every seed as having universal early failure.', '', 'Matrix validation degradation also repeats: the large post-warmup jumps occur at epochs 61, 46 and 77; final validation MSE is about 41, 41 and 64 times its best. Best matrix checkpoints are epochs 60, 39 and 71, all before the respective jump. Original seed 7 also has one threshold-crossing validation fluctuation at epoch 58, but recovers; the persistent matrix plateau is the distinguishing pattern. The later jumps cannot directly cause predictions from earlier saved checkpoints, and these logs do not identify a causal optimizer defect.', '', 'Conclusion: the current matrix pipeline is consistently worse on pooled test metrics and less stable under the fixed settings. It can help many batteries, so the evidence does not rule out a useful representation after controlled validation-led investigation. It does rule out claiming a robust improvement from the current implementation. No tuning, Zhang runs or FFT/no-FFT ablation was started.', '', 'The four additional runs consumed 33.68 minutes in training and 35.54 minutes in total arm time (including validation, evaluation and saving), excluding setup and report generation.', ''],
        '## Limitations and next steps','',
        'Three training seeds quantify initialization/order variability on one fixed held-out MIT date batch; they do not establish population significance. Matrix uses a larger first layer and 16-cycle history, so this is a pipeline comparison, not proof of an FFT-specific effect. No model tuning was performed. Any future tuning must use training/validation data, not these inspected test outcomes. Stop here; Zhang and ablations require a separate instruction.','',
        '## Artifacts','',
        'final_comparison.csv contains Wang aggregate means and sample SD only (two rows). wang_three_seed/ contains all six per-run metrics, 216 per-battery metric rows, paired improvements, per-seed diagnostics, aggregate battery MAE and plots. New checkpoints, histories and predictions are in W-O/seed_7, W-H/seed_7, W-O/seed_123 and W-H/seed_123. Source/configuration/manifests and diagnostic checks are in wang_shared_seed_7 and wang_shared_seed_123.','',
        'A setup-only variable-name error in the additional-seed runner was corrected before any training began; the failed setup directory is retained as wang_seed_7_setup_error. This did not change model, numerical settings or data protocol.','']
    (OUT/'ANALYSIS_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    (DEST/'input_sha256.json').write_text(json.dumps(hashes,indent=2))
    print(display.to_string(index=False));print(diag.to_string(index=False))

if __name__=='__main__':main()
