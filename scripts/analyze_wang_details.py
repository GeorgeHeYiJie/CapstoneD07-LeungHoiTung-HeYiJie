"""Read-only analysis of saved Wang predictions; no model imports or training."""
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
DEST=OUT/'wang_detailed'
DEST.mkdir(exist_ok=False)
inputs=[OUT/arm/'seed_42'/name for arm in ('W-O','W-H') for name in ('test_predictions.npz','loss_history.csv','metrics.json')]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
a={arm:dict(np.load(OUT/arm/'seed_42/test_predictions.npz',allow_pickle=False)) for arm in ('W-O','W-H')}
for key in ('y_true','battery_id','row_index'):assert np.array_equal(a['W-O'][key],a['W-H'][key])
d=pd.DataFrame(dict(battery=a['W-O']['battery_id'],cycle=a['W-O']['row_index'],soh=a['W-O']['y_true'].ravel().astype(float)))
assert not d.duplicated(['battery','cycle']).any()
lengths={b:len(pd.read_csv(ROOT/'data/MIT data/2018-04-12'/f'{b}.csv')) for b in d.battery.unique()}
d['trajectory_rows']=d.battery.map(lengths)
d['progress']=d.cycle/(d.trajectory_rows-1)
for arm,tag in [('W-O','o'),('W-H','h')]:
    d['p_'+tag]=a[arm]['y_pred'].ravel().astype(float)
    d['e_'+tag]=d['p_'+tag]-d.soh
    d['ae_'+tag]=d['e_'+tag].abs()
    d['se_'+tag]=d['e_'+tag]**2


def score(frame,tag):
    mse=float(frame['se_'+tag].mean())
    return dict(MAE=float(frame['ae_'+tag].mean()),MAPE_percent=float((frame['ae_'+tag]/frame.soh.abs()).mean()*100),
                RMSE=float(np.sqrt(mse)),MSE=mse,bias=float(frame['e_'+tag].mean()))


rows=[]
for b,g in d.groupby('battery',sort=True):
    o,h=score(g,'o'),score(g,'h')
    row=dict(battery=b,n=len(g),trajectory_rows=lengths[b],last_evaluated_cycle=int(g.cycle.max()),
             MAE_original=o['MAE'],MAE_matrix=h['MAE'],MAE_difference=h['MAE']-o['MAE'],
             MAE_improvement_percent=100*(o['MAE']-h['MAE'])/o['MAE'],
             MSE_original=o['MSE'],MSE_matrix=h['MSE'],bias_original=o['bias'],bias_matrix=h['bias'],
             excess_absolute_error=float((g.ae_h-g.ae_o).sum()),excess_squared_error=float((g.se_h-g.se_o).sum()))
    for part,mask in [('early',g.progress<.2),('late',g.progress>=.8)]:
        q=g[mask]
        row[part+'_MAE_delta']=float((q.ae_h-q.ae_o).mean())
        row[part+'_matrix_bias']=float(q.e_h.mean())
    rows.append(row)
rank=pd.DataFrame(rows).sort_values(['MAE_improvement_percent','battery'],ascending=[False,True]).reset_index(drop=True)
rank.insert(0,'rank',np.arange(1,len(rank)+1))
assert len(rank)==36
rank.to_csv(DEST/'battery_ranking.csv',index=False)
rank.head(10).to_csv(DEST/'top10_improved.csv',index=False)
rank.sort_values('MAE_improvement_percent').head(10).to_csv(DEST/'top10_worsened.csv',index=False)

# Error contribution is additive; relative per-battery improvement is not.
contrib=rank.sort_values('excess_absolute_error',ascending=False).copy()
pos=float(contrib.excess_absolute_error.clip(lower=0).sum())
net=float(contrib.excess_absolute_error.sum())
contrib['share_positive_excess_abs_percent']=100*contrib.excess_absolute_error.clip(lower=0)/pos
contrib['share_net_excess_abs_percent']=100*contrib.excess_absolute_error/net
contrib.to_csv(DEST/'error_contributions.csv',index=False)
sens=[]
for count in (0,1,3,5):
    removed=contrib.head(count).battery.tolist();q=d[~d.battery.isin(removed)]
    for tag in ('o','h'):
        sens.append(dict(excluded_count=count,excluded_batteries=';'.join(removed),arm=tag,n=len(q),**score(q,tag)))
pd.DataFrame(sens).to_csv(DEST/'posthoc_exclusion_sensitivity.csv',index=False)

def bins(frame,column,edges):
    result=[]
    groups=frame.groupby(pd.cut(frame[column],edges,right=False),observed=True)
    for interval,g in groups:
        row=dict(bin=str(interval),mid=float(g[column].mean()),n=len(g),batteries=g.battery.nunique())
        for tag in ('o','h'):
            row.update({tag+'_'+k:v for k,v in score(g,tag).items()})
            row[tag+'_macro_MAE']=float(g.groupby('battery')['ae_'+tag].mean().mean())
            row[tag+'_macro_bias']=float(g.groupby('battery')['e_'+tag].mean().mean())
        result.append(row)
    return pd.DataFrame(result)

specs=[('cycle',np.arange(0,2501,250),'Original CSV row index'),
       ('progress',np.linspace(0,1.000001,6),'Relative recorded trajectory position'),
       ('soh',[.8,.85,.9,.925,.95,.975,1.001],'True SOH')]
profiles={}
worst=contrib.iloc[0].battery
for col,edges,_ in specs:
    profiles[col]=bins(d,col,edges);profiles[col].to_csv(DEST/f'errors_by_{col}.csv',index=False)
    bins(d[d.battery!=worst],col,edges).to_csv(DEST/f'errors_by_{col}_without_largest_contributor.csv',index=False)

def save(fig,name):
    fig.tight_layout();fig.savefig(DEST/name,dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(10,11))
ax.barh(rank.battery.str.replace('2018-04-12_','',regex=False),rank.MAE_difference,
        color=np.where(rank.MAE_difference<0,'#287e70','#c35d45'))
ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
ax.set(xlabel='Matrix MAE - original MAE (SOH fraction; negative = better)',title='All 36 batteries, ranked by relative MAE improvement')
save(fig,'per_battery_mae_difference.png')

fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,col,label in zip(axs,['excess_absolute_error','excess_squared_error'],['absolute','squared']):
    q=rank.sort_values(col,ascending=False);vals=q[col].clip(lower=0)
    ax.plot(range(1,37),100*vals.cumsum()/vals.sum(),'o-',markersize=3)
    ax.set(xlabel='Batteries ranked by positive excess error',ylabel='Cumulative share (%)',title=f'Concentration of additional {label} error',ylim=(0,105));ax.grid(alpha=.2)
save(fig,'error_concentration.png')

for excluded,suffix in [(None,''),(worst,'_without_largest_contributor')]:
    frame=d if excluded is None else d[d.battery!=excluded]
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for j,(col,edges,label) in enumerate(specs):
        q=bins(frame,col,edges)
        for tag,name,c in [('o','Original','#296a9c'),('h','Matrix','#d17928')]:
            axs[0,j].plot(q.mid,q[tag+'_MAE'],'o-',label=name,color=c)
            axs[1,j].plot(q.mid,q[tag+'_bias'],'o-',label=name,color=c)
        for i in range(2):axs[i,j].set_xlabel(label);axs[i,j].grid(alpha=.2);axs[i,j].legend()
        axs[0,j].set_ylabel('MAE (SOH fraction)');axs[1,j].set_ylabel('Signed bias: prediction - true SOH');axs[1,j].axhline(0,c='black',lw=.7)
    fig.suptitle('Pooled binned errors'+(f' — diagnostic exclusion: {excluded}' if excluded else ' — all test batteries'))
    save(fig,'error_profiles'+suffix+'.png')

fig,axs=plt.subplots(1,2,figsize=(11,4))
axs[0].scatter(rank.trajectory_rows,rank.MAE_improvement_percent)
axs[0].set(xlabel='Recorded trajectory length (CSV rows)',ylabel='MAE improvement (%)')
axs[1].scatter(rank.trajectory_rows,rank.MAE_difference)
axs[1].set(xlabel='Recorded trajectory length (CSV rows)',ylabel='Matrix MAE - original MAE')
for ax in axs:ax.axhline(0,c='black',lw=.7);ax.grid(alpha=.2)
save(fig,'trajectory_length_vs_error.png')

selected=rank.head(3).battery.tolist()+rank.tail(3).iloc[::-1].battery.tolist()
fig,axs=plt.subplots(3,2,figsize=(13,10))
for ax,b in zip(axs.T.flat,selected):
    q=d[d.battery==b].sort_values('cycle')
    ax.plot(q.cycle,q.soh,label='True',c='black')
    ax.plot(q.cycle,q.p_o,label='Original',alpha=.8);ax.plot(q.cycle,q.p_h,label='Matrix',alpha=.8)
    ax.set(title=b.replace('2018-04-12_',''),xlabel='Original row index',ylabel='SOH');ax.legend(fontsize=8)
fig.suptitle('Left: three largest MAE improvements; right: three largest deteriorations')
save(fig,'selected_battery_trajectories.png')

hist={arm:pd.read_csv(OUT/arm/'seed_42/loss_history.csv').set_index('epoch') for arm in ('W-O','W-H')}
ho,hh=hist['W-O'],hist['W-H']
assert np.array_equal(ho.lr_used,hh.lr_used)
hh.loc[55:70].to_csv(DEST/'epoch55_70_spike.csv')
spikes=hh[(hh.index>30)&(hh.validation_MSE/hh.validation_MSE.shift(1)>5)].index.tolist()
fig,axs=plt.subplots(2,2,figsize=(12,7))
for arm,q in hist.items():
    axs[0,0].semilogy(q.index,q.validation_MSE,label=arm)
    axs[0,1].semilogy(q.index,q.data,label=arm)
for col,factor in [('data',1),('pde',.5),('physics',.01)]:
    axs[1,0].semilogy(hh.loc[50:80].index,hh.loc[50:80,col]*factor,label=f'{factor:g} x {col}')
axs[1,1].plot(hh.index,hh.lr_used,label='Solution Adam, both arms');axs[1,1].axhline(.001,c='grey',ls='--',label='Dynamics Adam')
for ax in axs.flat:ax.axvline(61,c='red',ls=':',label='Epoch 61');ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
axs[0,0].set_title('Validation MSE');axs[0,1].set_title('Training data loss');axs[1,0].set_title('Matrix weighted loss components (zoom)');axs[1,1].set_title('Learning-rate schedule: no jump at epoch 61')
save(fig,'epoch61_loss_and_lr.png')

def table(frame,cols):
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join([' --- ']*len(cols))+'|']
    for _,r in frame.iterrows():
        vals=[]
        for c in cols:
            v=r[c];vals.append(f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

summary=dict(total_samples=len(d),batteries=len(rank),improved=int((rank.MAE_difference<0).sum()),worsened=int((rank.MAE_difference>0).sum()),
    median_improvement=float(rank.MAE_improvement_percent.median()),largest_contributor=worst,
    top1_positive_excess_share=float(contrib.iloc[0].share_positive_excess_abs_percent),
    top3_positive_excess_share=float(contrib.head(3).share_positive_excess_abs_percent.sum()),
    top1_net_excess_share=float(contrib.iloc[0].share_net_excess_abs_percent),
    length_spearman_improvement=float(rank.trajectory_rows.corr(rank.MAE_improvement_percent,method='spearman')),
    length_spearman_delta=float(rank.trajectory_rows.corr(rank.MAE_difference,method='spearman')),
    late_vs_early_delta_worse_count=int((rank.late_MAE_delta>rank.early_MAE_delta).sum()),
    postwarmup_large_val_jump_epochs=spikes,
    data_spike_factor=float(hh.loc[61,'data']/hh.loc[60,'data']),
    val_spike_factor=float(hh.loc[61,'validation_MSE']/hh.loc[60,'validation_MSE']),
    best_to_final_val_ratio=float(hh.loc[200,'validation_MSE']/hh.loc[60,'validation_MSE']))
(DEST/'statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
cols=['rank','battery','n','MAE_original','MAE_matrix','MAE_difference','MAE_improvement_percent']
lines=['# Detailed Wang original vs matrix analysis','',
'Date: 2026-10-14. Read-only analysis of existing seed-42 predictions and 200-epoch loss histories. No model changes, inference, retraining, or Zhang experiments.','',
'## Main finding','',
f"The matrix model is **not broadly worse across batteries**: {summary['improved']}/36 improve in MAE and {summary['worsened']}/36 worsen; median battery-level relative improvement is {summary['median_improvement']:.2f}%. Its pooled performance is worse, with a concentrated tail of battery-specific failures and a separate training-instability event. The representation is potentially useful, but the present implementation needs validation-led investigation before any tuning claim.",
'','This is one seed, one held-out date batch, and two different-capacity/history pipelines. It neither establishes statistical superiority nor identifies FFT as the cause.','',
'## Definitions and evidence','',
'Inputs: W-O/W-H seed_42/test_predictions.npz, loss_history.csv and metrics.json. Arrays are joined by identical battery_id/row_index/true-label sequences; all 29,083 keys are unique. Models were selected separately by validation, at epochs 111 (original) and 60 (matrix). SOH is capacity/1.1 as a fraction. MAE difference = matrix MAE - original MAE (negative is better). Ranking uses 100*(original MAE-matrix MAE)/original MAE, descending; it is not a ranking by absolute error contribution.','',
'Trajectory length is the number of recorded CSV rows, not a verified end-of-life measurement. Relative position = original row index/(recorded rows-1), used only for retrospective analysis, never as a new model input. Histories may have different sampling/termination conditions.','',
'## Top 10 improved most','',table(rank.head(10),cols),'','## Top 10 worsened most','',table(rank.sort_values('MAE_improvement_percent').head(10),cols),
'','## All 36 ranked by MAE improvement','',table(rank,cols),'',
'![Per-battery MAE difference](wang_detailed/per_battery_mae_difference.png)','',
'## Are the pooled errors driven by a small subset?','',
f"Largest contributor to excess absolute error: **{worst}**, accounting for {summary['top1_positive_excess_share']:.1f}% of positive excess absolute error. The top three account for {summary['top3_positive_excess_share']:.1f}%. The largest battery contributes {summary['top1_net_excess_share']:.1f}% of NET excess error; a share over 100% is possible because other batteries offset that loss.",
'','Contribution = sum over a battery of (absolute matrix error - absolute original error). This equals sample count times its MAE difference. Squared-error concentration is shown separately. Percentage improvements on a low-error battery can be very large without dominating the total.','',
table(contrib.head(5),['battery','n','excess_absolute_error','share_positive_excess_abs_percent','share_net_excess_abs_percent','excess_squared_error']),
'','![Error concentration](wang_detailed/error_concentration.png)','',
'### Post-hoc exclusion sensitivity — diagnostic only','',
'Remove the top contributors to excess absolute error from BOTH arms, using the same remaining samples. These are deliberately outcome-selected exclusions, not replacement headline results and not permission to remove difficult batteries from future evaluation. o=original; h=matrix.','',
table(pd.DataFrame(sens),['excluded_count','arm','n','MAE','MAPE_percent','RMSE','MSE']),
'','## Error versus cycle index, SOH and recorded length','',
'Plots show pooled MAE and signed bias (prediction minus true SOH); negative bias means underprediction. Bins are descriptive, not new training thresholds. Counts and equal-battery-within-bin averages are included in the CSVs to expose unequal trajectory coverage. Very late absolute-cycle bins contain few long batteries; SOH, length and age are correlated, so these curves do not prove causation.','',
'![Binned error profiles](wang_detailed/error_profiles.png)','',
table(profiles['progress'],['bin','n','batteries','o_MAE','h_MAE','o_bias','h_bias']),
'','The pattern is mixed: matrix MAE is worse in the first 20% of the recorded trajectory for **34/36 batteries**, but in the final 20% for only **13/36**. Pooled early MAE rises from 0.003219 to 0.009478, with positive matrix bias +0.007281 (overprediction). This broad early bias remains after excluding battery 33. In the final 20%, pooled MAE is nearly tied (0.010236 original versus 0.010408 matrix); excluding battery 33 diagnostically changes this to 0.010903 versus 0.007192. Thus the catastrophic tail is concentrated, while an early/high-SOH weakness is widespread.\n\nThere is no uniform low-SOH failure: matrix MAE improves in the 0.80–0.85 band but worsens in 0.85–0.90. The lowest band contains only 307 samples from 24 batteries; its improved MAE does not imply improved RMSE. At high SOH (0.975 and above), matrix MAE rises from 0.001719 to 0.008980, with positive bias +0.008354. Recorded length shows only weak correlation with the improvement, so longer trajectories alone do not explain the failures.','','### SOH bands','',table(profiles['soh'],['bin','n','batteries','o_MAE','h_MAE','o_bias','h_bias']),
'',f"Within-battery late-vs-early check: {summary['late_vs_early_delta_worse_count']}/36 batteries have a greater matrix-minus-original MAE difference in the last 20% than in the first 20% of their recorded trajectory. This complements the pooled curves; it does not equate late-recording with a common physical age.",
'',f"Battery-level Spearman correlation of length with MAE improvement: {summary['length_spearman_improvement']:.3f}; with MAE difference: {summary['length_spearman_delta']:.3f}. This is descriptive across 36 correlated-condition batteries, not a hypothesis-test result.",
'','![Length relationship](wang_detailed/trajectory_length_vs_error.png)','',
f'Profiles with only {worst} excluded diagnose how strongly it affects the apparent low-SOH/late-cycle pattern. They do not change the official results.',
'','![Profiles without largest contributor](wang_detailed/error_profiles_without_largest_contributor.png)','',
'![Selected trajectories](wang_detailed/selected_battery_trajectories.png)','',
'## Epoch-61 spike: what the saved evidence supports','',
table(hh.loc[58:64].reset_index(),['epoch','lr_used','data','pde','physics','validation_MSE']),
'',f"Epoch 61 training data loss rises {summary['data_spike_factor']:.1f}x versus epoch 60; validation MSE rises {summary['val_spike_factor']:.1f}x. After warmup (epochs >30), validation jumps greater than 5x over the previous epoch occur at {spikes}. This declared descriptive threshold separates large shocks from ordinary fluctuations.",
'',f"It is an isolated large onset, **not an isolated harmless one-epoch outlier**: training data loss is still high at epoch 62, and validation settles on a worse plateau. Final validation MSE is {summary['best_to_final_val_ratio']:.1f}x the best at epoch 60; it never recovers that best during epochs 61–200.",
'','The solution LR is smoothly decreasing near the event, not being increased or restarted. The warmup/peak transition occurs around epochs 30–32, not 61. Both arms have identical saved LR sequences; dynamics LR remains 0.001. Adam runs continuously without an epoch-61 reset, gradient clipping or special branch in the saved runner. The relatively high LR, large input layer, dropout, minibatch order and Adam dynamics are plausible interacting factors, but the logs cannot identify which caused the jump.',
'','The checkpoints save model states, not Adam moments or per-step gradients. Epoch-aggregated losses cannot locate the offending minibatch or distinguish a large update from input/batch effects. Do not call this a confirmed optimizer bug. A later authorized diagnosis could log gradient/update norms around a replay, but no replay was performed here.',
'','Crucially, the matrix test predictions analyzed here come from **epoch 60, before the spike**. The later spike therefore cannot directly cause these saved test errors. It prevented a demonstrated later recovery in this run, but is a separate warning about training stability. Its association with validation degradation is direct; its causal explanation is unproven.',
'','![Loss spike and LR](wang_detailed/epoch61_loss_and_lr.png)','',
'## Recommendation','',
'Classify this as **potentially useful, with concentrated severe failures, widespread early/high-SOH bias, and training instability requiring investigation**, rather than broadly inferior on every battery. First inspect the largest-error trajectories and validation-side analogues; then, if authorized, design a controlled training-stability/representation ablation using validation only. Do not tune to the already-inspected test batteries or discard them. No model changes or further training are recommended as an automatic continuation of this analysis.','',
'## Files','',
'wang_detailed/ contains full ranking, top/bottom tens, additive error contributions, exclusion sensitivity, binned errors (including battery-balanced columns), epoch55–70 history, statistics and seven diagnostic plot families. Original prediction/checkpoint/metrics files remain unchanged.','']
(OUT/'WANG_DETAILED_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
(DEST/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('Sensitivity:',pd.DataFrame(sens).to_string(index=False))
print('Progress:',profiles['progress'].to_string(index=False))
print('SOH:',profiles['soh'].to_string(index=False))
