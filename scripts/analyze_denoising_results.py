"""Analyze saved paired offline-denoising predictions only; never trains."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_wang_comparison import metrics,write_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/denoising_analysis'
METRICS=['MAE','MAPE_percent','RMSE','MSE']


def improvement(a,b):return 100*(a-b)/a if a else None


def table(frame):
    cols=list(frame.columns)
    rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.7g}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(rows)


def analyze(test):
    folder=OUT/test
    names=['wang','zhang'] if test=='test1' else ['combined']
    allmetrics=[];comparisons=[];per=[];early=[];early_overall=[];outliers=[];stability=[];runtimes=[]
    configs=[json.loads((folder/c/'configuration.json').read_text()) for c in ['original','denoised']]
    assert {k:v for k,v in configs[0].items() if k!='condition'}=={k:v for k,v in configs[1].items() if k!='condition'}
    initials=[torch.load(folder/c/'initial_state.pt',weights_only=True) for c in ['original','denoised']]
    for name in names:
        for sa,sb in zip(initials[0][name],initials[1][name]):
            assert sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)
    text=[f'# {test.upper()}: OFFLINE original versus denoised MIT results','',
        '**Offline only.** Centered smoothing and full-trajectory pooling/interpolation use future-cycle information. These results do not establish real-time deployment performance.','',
        'One seed (42), 200 epochs per condition, no tuning or early stopping. The same 71/18/36 train/validation/test batteries and 29,083 held-out test samples are used. Each predictor is selected solely by its lowest full-validation SOH MSE. Labels and row IDs match exactly.','',
        'Train-only scalers are fitted separately using the same procedure for each feature condition. No outlier filtering or clipping changes the evaluated rows. SOH is capacity/1.1 in fraction units. MAE/RMSE use fractional SOH; MSE uses its square; MAPE uses true SOH as denominator and is a percentage. Overall metrics are sample-weighted.','',
        'The comparison is within this test; the other test is not its baseline. All smoke results are excluded.']
    for name in names:
        packs={c:np.load(folder/c/(name+'_test_predictions.npz')) for c in ['original','denoised']}
        a,b=packs['original'],packs['denoised']
        for k in ['y_true','battery_id','row_index','trajectory_length']:assert np.array_equal(a[k],b[k]),k
        y=a['y_true'].ravel();p=a['y_pred'].ravel();q=b['y_pred'].ravel();ids=a['battery_id'];rows=a['row_index'];lengths=a['trajectory_length']
        assert len(y)==29083 and len(np.unique(ids))==36
        original=metrics(y,p);denoised=metrics(y,q)
        records={}
        for c,metric in [('original',original),('denoised',denoised)]:
            record=next(x for x in json.loads((folder/c/'metrics.json').read_text()) if x['predictor']==name)
            for k in METRICS:assert np.isclose(record[k],metric[k],rtol=1e-12)
            records[c]=record;allmetrics.append(record)
            runtimes.append({k:record[k] for k in ['test','condition','predictor','seed','training_seconds','validation_seconds','test_inference_seconds','total_run_seconds','best_epoch','stopping_epoch','checkpoint']})
            hist=pd.read_csv(folder/c/'loss_history.csv');v=hist[name+'_validation_MSE'].to_numpy()
            jumps=(np.flatnonzero((v[1:]>5*v[:-1]) & (hist.epoch.to_numpy()[1:]>30))+1)
            stability.append(dict(predictor=name,condition=c,best_epoch=record['best_epoch'],best_validation_MSE=record['best_validation_MSE'],
                final_validation_MSE=v[-1],final_to_best_ratio=v[-1]/v.min(),jumps_over_5x_after_epoch30=';'.join(str(int(hist.epoch.iloc[j])) for j in jumps),
                max_validation_MSE=float(v.max())))
        for k in METRICS:
            comparisons.append(dict(predictor=name,metric=k,original=original[k],denoised=denoised[k],
                denoised_minus_original=denoised[k]-original[k],absolute_difference=abs(denoised[k]-original[k]),
                improvement_percent=improvement(original[k],denoised[k])))
        batteryrows=[]
        for bid in np.unique(ids):
            mask=ids==bid;ma=metrics(y[mask],p[mask]);mb=metrics(y[mask],q[mask])
            r=dict(predictor=name,battery_id=bid,n=int(mask.sum()))
            for k in METRICS:r.update({k+'_original':ma[k],k+'_denoised':mb[k],k+'_difference':mb[k]-ma[k],k+'_improvement_percent':improvement(ma[k],mb[k])})
            r['excess_absolute_error_sum']=float(np.abs(q[mask]-y[mask]).sum()-np.abs(p[mask]-y[mask]).sum())
            r['denoised_squared_error_share_percent']=float(100*np.square(q[mask]-y[mask]).sum()/np.square(q-y).sum())
            batteryrows.append(r);per.append(r)
            for cond,pred in [('original',p),('denoised',q)]:
                positions=np.flatnonzero(mask);worst=positions[np.argmax(np.abs(pred[mask]-y[mask]))]
                outliers.append(dict(predictor=name,condition=cond,battery_id=bid,MAE=metrics(y[mask],pred[mask])['MAE'],
                    max_absolute_error=float(abs(pred[worst]-y[worst])),worst_row=int(rows[worst]),true_SOH=float(y[worst]),predicted_SOH=float(pred[worst]),
                    outside_0_1_count=int(((pred[mask]<0)|(pred[mask]>1)).sum())))
        br=pd.DataFrame(batteryrows);improved=int((br.MAE_difference<0).sum());worse=int((br.MAE_difference>0).sum())
        positive=br.excess_absolute_error_sum.clip(lower=0);top=br.sort_values('excess_absolute_error_sum',ascending=False).head(3)
        share=100*top.excess_absolute_error_sum.clip(lower=0).sum()/positive.sum() if positive.sum() else 0.
        gains=(-br.excess_absolute_error_sum).clip(lower=0)
        gain_share=100*gains.nlargest(3).sum()/gains.sum() if gains.sum() else 0.
        early_mask=rows/np.maximum(lengths-1,1)<.2
        ea=metrics(y[early_mask],p[early_mask]);eb=metrics(y[early_mask],q[early_mask]);early_worse=0
        for k in METRICS:early_overall.append(dict(predictor=name,metric=k,n=int(early_mask.sum()),original=ea[k],denoised=eb[k],difference=eb[k]-ea[k],improvement_percent=improvement(ea[k],eb[k])))
        for bid in np.unique(ids):
            mask=(ids==bid)&early_mask
            aa=metrics(y[mask],p[mask]);bb=metrics(y[mask],q[mask]);early_worse+=bb['MAE']>aa['MAE']
            early.append(dict(predictor=name,battery_id=bid,n=int(mask.sum()),original_MAE=aa['MAE'],denoised_MAE=bb['MAE'],
                improvement_percent=improvement(aa['MAE'],bb['MAE']),original_bias=float(np.mean(p[mask]-y[mask])),denoised_bias=float(np.mean(q[mask]-y[mask]))))
        text += ['',f'## {name}: numerical comparison','',table(pd.DataFrame([r for r in comparisons if r['predictor']==name]).drop(columns='predictor')),
            '', 'Improvement = (original error − denoised error) / original error × 100. Positive is better; negative is worse. Signed difference is denoised minus original; absolute_difference is its magnitude.',
            '',f'Battery MAE: **{improved}/36 improved; {worse}/36 worsened**. Macro-average battery MAE: {br.MAE_original.mean():.8g} original versus {br.MAE_denoised.mean():.8g} denoised.',
            f'The three largest positive excess-absolute-error contributors account for {share:.2f}% of total positive excess error among worsened batteries. This measures concentration of worsening, not net overall improvement.',
            f'The three largest absolute-error reductions account for {gain_share:.2f}% of total positive error reductions among improved batteries. Median battery MAE improvement is {br.MAE_improvement_percent.median():+.2f}%. {"A majority of batteries improve" if improved>18 else "Improvement is not a majority-battery result"}; {"more than half of the positive gain is concentrated in three batteries" if gain_share>50 else "the three largest contributors account for no more than half of the positive gain"}. These are descriptive concentration measures, not significance tests.',
            '', 'Largest MAE improvements:', '',table(br.sort_values('MAE_improvement_percent',ascending=False)[['battery_id','MAE_original','MAE_denoised','MAE_improvement_percent']].head(10)),
            '', 'Worst MAE changes:', '',table(br.sort_values('MAE_improvement_percent')[['battery_id','MAE_original','MAE_denoised','MAE_improvement_percent']].head(10)),
            '',f'Early-cycle pooled MAE: {ea["MAE"]:.8g} original versus {eb["MAE"]:.8g} denoised ({improvement(ea["MAE"],eb["MAE"]):+.2f}% improvement). Early MAE worsened on {early_worse}/36 batteries.',
            f'Early signed bias: {np.mean(p[early_mask]-y[early_mask]):+.8g} original versus {np.mean(q[early_mask]-y[early_mask]):+.8g} denoised. Positive bias means overprediction.',
            '', 'Training/validation stability:', '',table(pd.DataFrame([r for r in stability if r['predictor']==name])),
            '', 'Runtime and selection:', '',table(pd.DataFrame([r for r in runtimes if r['predictor']==name]).drop(columns=['test','predictor'])),
            '', 'Largest individual denoised prediction errors:', '',table(pd.DataFrame([r for r in outliers if r['predictor']==name and r['condition']=='denoised']).sort_values('max_absolute_error',ascending=False).head(5)),
            '',f'Cautious finding: denoising {"reduced" if denoised["MAE"]<original["MAE"] else "increased"} pooled MAE by {abs(improvement(original["MAE"],denoised["MAE"])):.2f}% in this seed. The battery counts, macro-average and outlier concentration above determine whether this reflects broad or concentrated changes. This is not a statistical-significance claim and does not establish an online benefit.']
        # Shared metric and matched prediction figures.
        fig,axs=plt.subplots(1,4,figsize=(13,3.6))
        for ax,k in zip(axs,METRICS):
            ax.bar(['Original','Denoised'],[original[k],denoised[k]],color=['#286998','#d1822b'])
            ax.set_title(f'{k}\n{improvement(original[k],denoised[k]):+.2f}% improvement')
        fig.suptitle(f'{test} / {name} / seed 42 / OFFLINE');fig.tight_layout();fig.savefig(folder/(name+'_metrics.png'),dpi=160);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,label,pred in zip(axs,['Original','Denoised'],[p,q]):
            ax.scatter(y,pred,s=2,alpha=.2);lo=min(y.min(),pred.min());hi=max(y.max(),pred.max());ax.plot([lo,hi],[lo,hi],'k--')
            ax.set(title=label,xlabel='True SOH',ylabel='Predicted SOH')
        fig.tight_layout();fig.savefig(folder/(name+'_prediction_vs_true.png'),dpi=160);plt.close(fig)
        ordered=br.sort_values('MAE_difference');fig,ax=plt.subplots(figsize=(12,5));diff=ordered.MAE_difference
        ax.bar(np.arange(36),diff,color=np.where(diff<0,'#286998','#c75245'));ax.axhline(0,c='black',lw=.7)
        ax.set_xticks(np.arange(36),[s.split('battery-')[-1] for s in ordered.battery_id],rotation=90)
        ax.set(xlabel='Test battery number',ylabel='Denoised MAE − original MAE',title=f'{name}: negative means improved');fig.tight_layout();fig.savefig(folder/(name+'_per_battery_MAE_difference.png'),dpi=160);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for cond,pred in [('original',p),('denoised',q)]:
            rel=rows/np.maximum(lengths-1,1);positions=[];errors=[]
            for lower in np.arange(0,1,.1):
                mask=(rel>=lower)&(rel<lower+.1);positions.append(lower+.05);errors.append(float(np.abs(pred[mask]-y[mask]).mean()))
            axs[0].plot(positions,errors,'o-',label=cond)
            hist=pd.read_csv(folder/cond/'loss_history.csv');axs[1].semilogy(hist.epoch,hist[name+'_validation_MSE'],label=cond)
        axs[0].set(xlabel='Relative recorded trajectory position',ylabel='MAE',title='Early to late cycle errors')
        axs[1].set(xlabel='Epoch',ylabel='Validation MSE',title='Checkpoint-selection trace')
        for ax in axs:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(folder/(name+'_early_and_validation.png'),dpi=160);plt.close(fig)
        columns=['data','pde','physics','wang_total'] if name=='wang' else (['intra','delta','zhang_total'] if name=='zhang' else ['data','pde','physics','total'])
        fig,axs=plt.subplots(1,len(columns),figsize=(3.2*len(columns),3.5))
        for ax,col in zip(np.atleast_1d(axs),columns):
            for cond in ['original','denoised']:
                hist=pd.read_csv(folder/cond/'loss_history.csv');ax.plot(hist.epoch,hist[col],label=cond)
            ax.set(title=col,xlabel='Epoch');ax.set_yscale('symlog',linthresh=1e-6);ax.legend(fontsize=8)
        fig.suptitle(f'{name}: training loss behaviour');fig.tight_layout();fig.savefig(folder/(name+'_training_losses.png'),dpi=160);plt.close(fig)
        traj=folder/'per_battery_plots';traj.mkdir(exist_ok=True)
        for bid in np.unique(ids):
            mask=ids==bid;fig,ax=plt.subplots(figsize=(8,4));ax.plot(rows[mask],y[mask],c='black',label='True')
            ax.plot(rows[mask],p[mask],label='Original');ax.plot(rows[mask],q[mask],label='Denoised')
            ax.set(title=f'{name}: {bid}',xlabel='Original CSV row',ylabel='SOH fraction');ax.legend();fig.tight_layout();fig.savefig(traj/(name+'_'+bid+'.png'),dpi=130);plt.close(fig)
    write_csv(folder/'comparison_metrics.csv',comparisons);write_csv(folder/'run_metrics.csv',allmetrics)
    write_csv(folder/'per_battery_metrics.csv',per);write_csv(folder/'early_cycle_metrics.csv',early)
    write_csv(folder/'early_cycle_overall_metrics.csv',early_overall)
    write_csv(folder/'outlier_metrics.csv',outliers);write_csv(folder/'stability.csv',stability);write_csv(folder/'runtime.csv',runtimes)
    text += ['', 'Runtime for Test 1 is the shared Wang+Zhang condition runtime, not a separately timed training cost for each predictor. Matrix-cache preparation is recorded under prepared/ and excluded from training runtime. Validation stability alone does not establish test robustness. Any future tuning must use training/validation data, not the inspected test outcomes.',
             '', 'See comparison_metrics.csv, per_battery_metrics.csv, early_cycle_metrics.csv, outlier_metrics.csv, stability.csv and runtime.csv for complete numerical results. Predictions/checkpoints and full training-loss histories remain under original/ and denoised/.']
    (folder/'RESULT_ANALYSIS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    (folder/'analysis_verification.json').write_text(json.dumps(dict(initial_weights_exact=True,paired_configurations_exact=True,
        paired_test_ids_rows_labels_exact=True,metrics_recalculated=True,test_samples=29083,batteries=36,offline=True),indent=2)+'\n')
    if all((OUT/t/'comparison_metrics.csv').exists() for t in ['test1','test2']):
        joined=pd.concat([pd.read_csv(OUT/t/'comparison_metrics.csv').assign(test=t) for t in ['test1','test2']],ignore_index=True)
        joined.to_csv(OUT/'final_comparison.csv',index=False)
        lines=['# Offline denoising comparison: completed results','',
               '**OFFLINE ONLY:** the denoiser uses future-cycle information. These results do not support a real-time deployment claim.','',
               'Seed 42; 200 epochs per condition; 71/18/36 frozen MIT batteries; 29,083 matching test samples; condition-specific training-only normalization; no outlier filtering; independent validation-MSE checkpoint selection.','',
               'The comparisons below are original versus denoised within each test/predictor, not Test 1 versus Test 2. Positive improvement means lower denoised error.','',table(joined),
               '', 'Detailed battery, early-cycle, outlier, stability, runtime and checkpoint findings:',
               '', '- [Test 1 result analysis](test1/RESULT_ANALYSIS.md)', '- [Test 2 result analysis](test2/RESULT_ANALYSIS.md)',
               '', 'Each test retains its prediction arrays, best checkpoints, training histories and verification records. One seed and one held-out MIT batch do not establish statistical significance or generalization. No tuning was performed.']
        (OUT/'ANALYSIS_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(f'{test}: analysis complete, {len(comparisons)} metric comparisons',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',choices=['test1','test2'],required=True);args=p.parse_args();analyze(args.test)
