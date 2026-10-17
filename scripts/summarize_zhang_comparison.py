"""Analyze completed Zhang seed-42 predictions; no training or model imports."""
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
DEST=OUT/'zhang_analysis'
ARMS=['Z-O','Z-H']
METRICS=['MAE','MAPE_percent','RMSE','MSE']

def scores(y,p):
    e=p-y
    return dict(MAE=np.abs(e).mean(),MAPE_percent=100*np.abs(e/y).mean(),RMSE=np.sqrt((e**2).mean()),MSE=(e**2).mean(),bias=e.mean())

def table(df):
    lines=['| '+' | '.join(df.columns)+' |','|'+'|'.join([' --- ']*len(df.columns))+'|']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.7g}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)

def main():
    files=[OUT/a/'seed_42'/n for a in ARMS for n in ['metrics.json','test_predictions.npz','loss_history.csv']]
    assert all(p.exists() for p in files)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    DEST.mkdir(exist_ok=False)
    baseline=dict(np.load(OUT/'W-O/seed_42/test_predictions.npz'))
    data=pd.DataFrame(dict(battery=baseline['battery_id'],cycle=baseline['row_index'],y=baseline['y_true'].ravel().astype(float)))
    assert not data.duplicated(['battery','cycle']).any() and len(data)==29083
    lengths={p.stem:len(pd.read_csv(p)) for p in (ROOT/'data/MIT data/2018-04-12').glob('*.csv')}
    data['progress']=data.cycle/(data.battery.map(lengths)-1)
    runs=[];histories={};per=[];stability=[]
    for arm in ARMS:
        folder=OUT/arm/'seed_42'
        pred=dict(np.load(folder/'test_predictions.npz'))
        for k in ['y_true','battery_id','row_index']:assert np.array_equal(pred[k],baseline[k])
        p=pred['y_pred'].ravel().astype(float);data[arm]=p
        m=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        assert m['checkpoint_verified'] and m['stopping_epoch']==200 and m['seed']==42
        computed=scores(data.y.to_numpy(),p)
        for k in METRICS:assert np.isclose(m[k],computed[k],rtol=1e-12,atol=1e-14)
        runs.append(m)
        for b,g in data.groupby('battery'):
            early=g[g.progress<.2];e=g[arm]-g.y
            per.append(dict(arm=arm,seed=42,battery=b,n=len(g),**scores(g.y.to_numpy(),g[arm].to_numpy()),early_MAE=np.abs(early[arm]-early.y).mean(),early_bias=(early[arm]-early.y).mean(),max_absolute_error=np.abs(e).max()))
        q=pd.read_csv(folder/'loss_history.csv');histories[arm]=q
        assert len(q)==200 and q.epoch.tolist()==list(range(1,201))
        ratio=q.validation_MSE/q.validation_MSE.shift(1)
        stability.append(dict(arm=arm,best_epoch=m['best_epoch'],final_to_best_validation_ratio=m['final_validation_MSE']/m['best_validation_MSE'],postwarmup_jumps_over_5x=';'.join(map(str,q.loc[(q.epoch>30)&(ratio>5),'epoch'].tolist())),last20_validation_mean=q.validation_MSE.tail(20).mean(),last20_validation_std=q.validation_MSE.tail(20).std(ddof=1)))
    errors=data.copy()
    errors['matrix_error']=errors['Z-H']-errors.y
    errors['matrix_squared_error_share_percent']=100*errors.matrix_error**2/(errors.matrix_error**2).sum()
    errors.loc[errors.matrix_error.abs().sort_values(ascending=False).index].head(20).to_csv(DEST/'largest_sample_errors.csv',index=False)
    runs=pd.DataFrame(runs);per=pd.DataFrame(per);stability=pd.DataFrame(stability)
    runs.to_csv(DEST/'metrics_by_run.csv',index=False);per.to_csv(DEST/'per_battery_metrics.csv',index=False);stability.to_csv(DEST/'validation_behavior.csv',index=False)
    o=per[per.arm=='Z-O'].set_index('battery');h=per[per.arm=='Z-H'].set_index('battery')
    paired=pd.DataFrame(dict(n=o.n,MAE_original=o.MAE,MAE_matrix=h.MAE,MAE_difference=h.MAE-o.MAE,MAE_improvement_percent=100*(o.MAE-h.MAE)/o.MAE,early_MAE_original=o.early_MAE,early_MAE_matrix=h.early_MAE,matrix_bias=h.bias,max_absolute_error_matrix=h.max_absolute_error))
    paired['excess_absolute_error']=paired.n*paired.MAE_difference
    positive=paired.excess_absolute_error.clip(lower=0).sum()
    paired['positive_excess_share_percent']=100*paired.excess_absolute_error.clip(lower=0)/positive if positive>0 else 0
    paired['matrix_squared_error_share_percent']=100*(h.MSE*h.n)/(h.MSE*h.n).sum()
    paired['matrix_MAE_vs_median_battery']=h.MAE/h.MAE.median()
    paired=paired.sort_values('MAE_improvement_percent',ascending=False)
    paired.to_csv(DEST/'per_battery_comparison.csv')
    improvements=[]
    for k in METRICS:
        original=float(runs.loc[runs.arm=='Z-O',k].iloc[0]);matrix=float(runs.loc[runs.arm=='Z-H',k].iloc[0])
        improvements.append(dict(metric=k,original=original,matrix=matrix,improvement_percent=100*(original-matrix)/original))
    improvements=pd.DataFrame(improvements);improvements.to_csv(DEST/'improvements.csv',index=False)
    early=data[data.progress<.2]
    early_rows=[dict(arm=a,n=len(early),**scores(early.y.to_numpy(),early[a].to_numpy())) for a in ARMS]
    early_table=pd.DataFrame(early_rows);early_table.to_csv(DEST/'early_cycle_metrics.csv',index=False)
    profiles=[]
    for column,edges in [('progress',[0,.2,.4,.6,.8,1.000001]),('y',[.8,.85,.9,.925,.95,.975,1.001])]:
        for interval,g in data.groupby(pd.cut(data[column],edges,right=False),observed=True):
            for arm in ARMS:
                profiles.append(dict(axis=column,bin=str(interval),position=g[column].mean(),arm=arm,n=len(g),batteries=g.battery.nunique(),**scores(g.y.to_numpy(),g[arm].to_numpy())))
    profiles=pd.DataFrame(profiles);profiles.to_csv(DEST/'error_profiles.csv',index=False)
    diagnostics=dict(batteries_improved=int((paired.MAE_difference<0).sum()),batteries_worsened=int((paired.MAE_difference>0).sum()),early_batteries_worsened=int((paired.early_MAE_matrix>paired.early_MAE_original).sum()),largest_excess_battery=paired.excess_absolute_error.idxmax(),worst_matrix_MAE_battery=paired.MAE_matrix.idxmax(),matrix_batteries_over_5x_median_MAE=paired.index[paired.matrix_MAE_vs_median_battery>5].tolist(),test_samples=len(data))
    (DEST/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,11))
    ax.barh(paired.index.str.replace('2018-04-12_battery-','',regex=False),paired.MAE_difference,color=np.where(paired.MAE_difference<0,'#287b6c','#c45e41'))
    ax.invert_yaxis();ax.axvline(0,c='black',lw=.8);ax.set(xlabel='Matrix MAE - original MAE (negative is better)',ylabel='Battery number',title='Zhang seed 42: all 36 batteries, ranked by relative MAE improvement')
    fig.tight_layout();fig.savefig(DEST/'per_battery_mae_difference.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for arm,q in histories.items():
        axs[0,0].semilogy(q.epoch,q.validation_MSE,label=arm)
        for ax,k in zip([axs[0,1],axs[1,0],axs[1,1]],['total','intra','delta']):ax.semilogy(q.epoch,q[k],label=arm);ax.set_title(k+' training loss')
    axs[0,0].set_title('Validation mixed-prediction MSE')
    for ax in axs.flat:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(DEST/'training_validation.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for j,column in enumerate(['progress','y']):
        for arm in ARMS:
            q=profiles[(profiles.axis==column)&(profiles.arm==arm)]
            axs[0,j].plot(q.position,q.MAE,marker='o',label=arm)
            axs[1,j].plot(q.position,q.bias,marker='o',label=arm)
        for ax in axs[:,j]:ax.set_xlabel('Recorded trajectory position' if column=='progress' else 'True SOH');ax.legend()
        axs[0,j].set_ylabel('MAE');axs[1,j].set_ylabel('Signed bias');axs[1,j].axhline(0,c='black',lw=.8)
    fig.tight_layout();fig.savefig(DEST/'early_and_soh_profiles.png',dpi=150);plt.close(fig)
    selected=list(dict.fromkeys(['2018-04-12_battery-33',diagnostics['largest_excess_battery'],diagnostics['worst_matrix_MAE_battery']]))
    fig,axs=plt.subplots(1,len(selected),figsize=(6*len(selected),4),squeeze=False)
    for ax,b in zip(axs.flat,selected):
        g=data[data.battery==b];ax.plot(g.cycle,g.y,c='black',label='True')
        for arm in ARMS:ax.plot(g.cycle,g[arm],label=arm,alpha=.8)
        ax.set(title=b,xlabel='CSV row index',ylabel='SOH fraction');ax.legend()
    fig.tight_layout();fig.savefig(DEST/'outlier_trajectories.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,arm in zip(axs,ARMS):
        ax.scatter(data.y,data[arm],s=2,alpha=.2)
        lo=min(data.y.min(),data[arm].min());hi=max(data.y.max(),data[arm].max())
        ax.plot([lo,hi],[lo,hi],'k--');ax.set(title=arm,xlabel='True SOH',ylabel='Predicted SOH')
    fig.tight_layout();fig.savefig(DEST/'prediction_vs_true.png',dpi=150);plt.close(fig)
    # Preserve the prior Wang aggregate values exactly; single-seed SD is undefined.
    agg=pd.read_csv(OUT/'zhang_shared_seed_42/previous_final_comparison.csv')
    assert len(agg)==2 and set(agg.model)=={'Wang'}
    new=[]
    for r in runs.to_dict('records'):
        row=dict(model='Zhang-inspired SOH adaptation',arm=r['arm'],seeds='42',n_seeds=1,aggregation='single seed; SD undefined')
        for k in METRICS+['training_runtime_seconds','total_arm_seconds','best_epoch']:row[k+'_mean']=r[k];row[k+'_std']=np.nan
        new.append(row)
    final=pd.concat([agg,pd.DataFrame(new)],ignore_index=True)
    final['matrix_vs_original_improvement_percent_definition']='ratio of mean errors'
    for k in METRICS:
        final[k+'_improvement_percent']=np.nan
        final.loc[final.arm=='W-H',k+'_improvement_percent']=100*(agg.loc[agg.arm=='W-O',k+'_mean'].iloc[0]-agg.loc[agg.arm=='W-H',k+'_mean'].iloc[0])/agg.loc[agg.arm=='W-O',k+'_mean'].iloc[0]
        final.loc[final.arm=='Z-H',k+'_improvement_percent']=improvements.loc[improvements.metric==k,'improvement_percent'].iloc[0]
    final.to_csv(OUT/'final_comparison.csv',index=False)
    lines=['## Zhang seed-42 comparison','',
        'Only Z-O and Z-H were newly trained, 200 epochs each with Adam lr=0.001 and no scheduler/early stopping. Existing model classes and losses are unchanged. These are BatLiNet-inspired SOH adaptations, not a full reproduction of the Zhang lifetime model.','',
        'Frozen HD-SOH-v1 matches Wang: 71/18/36 batteries, capacity/1.1 targets, train-only min-max, same 29,083 test keys and labels, same standard metric definitions. Input shapes are [B,1,16,16,1] original and [B,2,16,8,8] matrix. Parameter counts remain 9,313 and 18,529 respectively. Thus this is a pipeline comparison with differing input channels, not an isolated FFT effect.','',
        'References use training batteries only. Training reference RNG=10042; each evaluation resets its independent RNG (validation 20042, test 30042), fixing two distinct references per sample across epochs and arms. Reference IDs, clamped rows and clamp counts are saved in zhang_shared_seed_42/. Training batch order and reference schedule are paired across arms. Checkpoints are selected using final mixed-prediction validation MSE only.','',
        table(improvements),'',
        'Positive improvement means matrix has lower error. MAE/RMSE use SOH fractions, MSE squared fractions, MAPE percent. Single-seed SD is undefined (blank in final_comparison.csv), not zero. Wang still has three seeds, so variability estimates are not comparable across methods.','',
        table(runs[['arm','training_runtime_seconds','validation_runtime_seconds','total_arm_seconds','inference_cold_seconds','inference_warm_seconds','best_epoch','stopping_epoch']]),'',
        'Inference timings include batch/reference preparation. The first test pass can include cache misses; a small diagnostic batch was already cached. Predictions and checkpoint reloads are verified exactly.','',
        f"Battery MAE improves on {diagnostics['batteries_improved']}/36 and worsens on {diagnostics['batteries_worsened']}/36. Largest excess-absolute-error contributor: {diagnostics['largest_excess_battery']}. Worst matrix MAE battery: {diagnostics['worst_matrix_MAE_battery']}.",'',
        '### Outlier concentration','',table(paired.sort_values('excess_absolute_error',ascending=False).head(5).reset_index()),'',
        'Battery 33 specifically:','',table(paired.loc[['2018-04-12_battery-33']].reset_index()),'',
        'Positive excess share is measured among worsened batteries; squared-error share is relative to the entire matrix test set. More than five times median battery MAE is a descriptive outlier flag, not a universal definition of catastrophe. Flagged batteries: '+str(diagnostics['matrix_batteries_over_5x_median_MAE'])+'. No battery is removed from official metrics.','',
        '![Per-battery MAE](zhang_analysis/per_battery_mae_difference.png)','',
        '![Outlier trajectories](zhang_analysis/outlier_trajectories.png)','',
        '### Early-cycle behavior','',table(early_table),'',
        f"Early MAE worsens on {diagnostics['early_batteries_worsened']}/36 batteries. Early is the first 20% of recorded rows, used only retrospectively for analysis. Positive bias is overprediction. Full error profiles by recorded position and SOH, with sample/battery counts, are saved in error_profiles.csv.",'',
        '![Early and SOH profiles](zhang_analysis/early_and_soh_profiles.png)','',
        '### Validation behavior','',table(stability),'',
        'The >5x jump threshold after epoch 30 is the same descriptive rule used for Wang. A transient fluctuation alone does not establish persistent instability. Final/best ratios and the full curves must also be inspected. Fixed reference selection prevents epoch-to-epoch reference resampling from confounding validation.','',
        '![Training and validation](zhang_analysis/training_validation.png)','',
        '![Prediction versus truth](zhang_analysis/prediction_vs_true.png)','',
        *['### Interpretation of Zhang seed 42', '', 'The matrix pipeline improves all four pooled metrics: MAE 19.15%, MAPE 20.05%, RMSE 12.35%, MSE 23.17%. Improvement is broad at the battery level (34/36), rather than caused by one lucky battery. Only batteries 38 and 39 worsen in full-trajectory MAE. This is encouraging for this seed, but does not establish multi-seed reliability or an FFT-specific benefit; input channels and parameter counts differ.', '', 'Battery 33 is not the Zhang failure case: its MAE falls from 0.007851 to 0.005534 (29.51% improvement), and it accounts for only 1.03% of matrix squared error. However, battery 39 has catastrophic late-sample overprediction. Its MAE rises from 0.033763 to 0.044774; it contributes 21.05% of total matrix squared error despite containing only 91/29,083 test samples. At original CSV row 90, true SOH is 0.802718, original predicts 1.322480 and matrix predicts 1.781437. Rows 89 and 90 alone contribute 18.52% of matrix squared error. A battery-average threshold (>5 times median battery MAE) misses this localized failure, so the empty flag list below must not be read as absence of catastrophic samples. Largest sample errors are saved separately in largest_sample_errors.csv.', '', 'Inspection of the existing features shows a sharp distribution shift in battery 39: the largest absolute train-normalized feature increases from about 1.18 at row 87 to 33.53, 66.44 and 65.80 at rows 88-90. This coincides with the prediction failures in both representations, but does not establish causation. The frozen protocol intentionally does not clip test inputs or remove outliers; no rows were removed and no model was changed.', '', 'Early-cycle weakness remains: pooled first-20%-trajectory MAE rises from 0.015983 to 0.017162 (7.37% worse), and 28/36 batteries worsen early. Both models overpredict in this region. Matrix gains are mainly in the middle/later recorded trajectory and lower-SOH bands; see the binned profiles rather than assuming uniform improvement.', '', "Training does not reproduce Wang matrix's persistent collapse in this run. Neither Zhang arm has a post-epoch-30 validation jump exceeding 5 times its previous value; smaller transient spikes occur, especially for matrix, and recover. Final/best validation MSE ratios are 1.114 (original) and 1.014 (matrix), versus about 41-64 for the Wang matrix runs. This is evidence of stable completion for seed 42, not proof of stability across seeds.", '', 'Best epochs are 72 (original) and 124 (matrix). Measured total-arm runtimes are 17.53 and 27.79 minutes, totaling 45.32 minutes; training alone totals 37.86 minutes. Setup and analysis time are outside those totals. No Wang training, further Zhang seeds, tuning or ablations were run.', '', 'Reference-row clamping is part of the frozen protocol: 9,144/27,170 validation reference assignments and 24,640/58,166 test reference assignments use the last available training-reference row. The assignments are identical across the two arms and do not consult evaluation targets, but row-index alignment is an approximation rather than physical-age matching.', '', 'Conclusion: Zhang matrix is promising in this one-seed pipeline comparison, with broad per-battery improvement and no persistent training collapse, while retaining early-SOH overprediction and a severe battery-39 tail failure. It is not yet a robust multi-seed result and does not show that FFT alone caused the gain.', ''],
        '### Artifacts and scope','',
        'New checkpoints, full histories and predictions: Z-O/seed_42 and Z-H/seed_42. Shared configuration/source snapshots/manifests/smoke checks: zhang_shared_seed_42. Per-run metrics, all 72 battery metric rows, paired improvements and plots: zhang_analysis. final_comparison.csv now contains Wang three-seed aggregates and Zhang one-seed values. No Wang training, architecture changes, hyperparameter tuning or ablation was performed. Stop after these two runs.','']
    prior=(OUT/'zhang_shared_seed_42/previous_analysis_summary.md').read_text(encoding='utf-8')
    # The full unedited historical Wang summary is retained in the shared snapshot.
    prior=prior.replace('# Wang MIT comparison: three seeds','# MIT comparison: Wang three seeds and Zhang seed 42',1)
    prior=prior.replace('No Zhang experiments and no FFT/no-FFT ablation were run.','At the Wang-stage completion, no Zhang experiments or ablations had been run; the newly completed Zhang stage is reported below.')
    prior=prior.replace('No tuning, Zhang runs or FFT/no-FFT ablation was started.','No tuning or ablation was performed; the subsequent Zhang stage is reported below.')
    prior=prior.replace('Stop here; Zhang and ablations require a separate instruction.','The subsequent authorized Zhang seed-42 stage is reported below; ablations remain unperformed.')
    prior=prior.replace('final_comparison.csv contains Wang aggregate means and sample SD only (two rows).','The previous two-row final_comparison.csv is preserved in zhang_shared_seed_42/previous_final_comparison.csv; the current table also includes Zhang.')
    (OUT/'ANALYSIS_SUMMARY.md').write_text(prior+'\n\n'+'\n'.join(lines),encoding='utf-8')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    (DEST/'input_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(improvements.to_string(index=False));print(json.dumps(diagnostics,indent=2));print(early_table.to_string(index=False));print(stability.to_string(index=False))

if __name__=='__main__':main()
