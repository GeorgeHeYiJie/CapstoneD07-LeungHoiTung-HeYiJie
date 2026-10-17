"""Run only W-O and W-H, one fresh seed each, under frozen HD-SOH-v1."""
import csv
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import high_dimension_comparison as h


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64).reshape(-1);p=np.asarray(p,dtype=np.float64).reshape(-1)
    d=p-y;mse=float(np.mean(d*d));mask=y!=0
    return dict(MAE=float(np.mean(abs(d))),MAPE_percent=float(100*np.mean(abs(d[mask]/y[mask]))) if mask.any() else None,
                RMSE=float(np.sqrt(mse)),MSE=mse,n=len(y))


def evaluate(model,data,split):
    keys=data.pairs[split];ys=[];ps=[];start=time.perf_counter()
    for i in range(0,len(keys),h.BATCH):
        y,p=model.predict(data,keys[i:i+h.BATCH],None);ys.append(y);ps.append(p)
    return np.concatenate(ys),np.concatenate(ps),time.perf_counter()-start


def load(model,path):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    for m,state in zip(model.modules,obj['states']):m.load_state_dict(state)
    return obj


def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(8)
    out=h.OUT;shared=out/'wang_shared';shared.mkdir(exist_ok=False)
    shutil.copy2(out/'ANALYSIS_SUMMARY.md',shared/'previous_preflight_summary.md')
    start=time.perf_counter();data=h.Data();data.write_records(shared)
    # Exact match with the previously measured and accepted protocol.
    for filename in ('split_manifest.json','scaler.json','dataset_manifest.json'):
        assert json.loads((shared/filename).read_text())==json.loads((out/'preflight'/filename).read_text())
    h.save_json(shared/'checks.json',data.checks())
    prep=time.perf_counter()-start
    for name in ('run_wang_comparison.py','high_dimension_comparison.py'):
        shutil.copy2(Path(__file__).parent/name,shared/name)
    h.save_json(shared/'configuration.json',dict(protocol='HD-SOH-v1',arms=['W-O','W-H'],seed=42,
        split_seed=420,batch_size=512,epochs=200,early_stopping=False,threads=8,
        checkpoint='each arm own lowest validation MSE',normalization='frozen train-only minmax',
        alpha=.5,beta=.01,learning_rate=dict(warmup=.002,base=.01,final=.0002,F=.001,warmup_epochs=30),
        shared_data_load_seconds=prep,environment=json.loads((out/'preflight/config.json').read_text())))
    # Bounded fresh one-batch update + checkpoint roundtrip before long runs.
    checks={}
    for name in ('W-O','W-H'):
        m=h.Arm(name);keys=data.pairs['train'][:h.BATCH]
        before=next(m.u.parameters()).detach().clone()
        loss=m.step(data,keys,np.random.default_rng(10042))
        assert not torch.equal(before,next(m.u.parameters()))
        y,p=m.predict(data,keys,None)
        cp=shared/(name+'_diagnostic.pt');torch.save(dict(states=m.states()),cp)
        load(m,cp);_,p2=m.predict(data,keys,None)
        assert np.array_equal(p,p2)
        checks[name]=dict(one_batch_loss=loss,parameters_updated=True,checkpoint_exact=True)
    h.save_json(shared/'runtime_checks.json',checks)
    results=[];predictions={};per_battery=[]
    for name in ('W-O','W-H'):
        dest=out/name/'seed_42';dest.mkdir(parents=True,exist_ok=False)
        model=h.Arm(name) # Fresh seed; never continues diagnostic weights.
        torch.save(dict(torch_rng=torch.get_rng_state(),states=model.states()),dest/'initial_state.pt')
        order_rng=np.random.default_rng(42);ref_rng=np.random.default_rng(10042)
        best=float('inf');history=[];train_seconds=0.;val_seconds=0.;arm_start=time.perf_counter()
        keys=data.pairs['train']
        print(f'{name}: starting 200 epochs; train={len(keys)}, validation={len(data.pairs["valid"])}, test={len(data.pairs["test"])}',flush=True)
        for epoch in range(1,201):
            order=order_rng.permutation(len(keys));losses=[];weights=[]
            lr=float(model.opts[0].param_groups[0]['lr']);t=time.perf_counter()
            for j in range(0,len(keys),h.BATCH):
                batch=keys[order[j:j+h.BATCH]]
                losses.append(model.step(data,batch,ref_rng));weights.append(len(batch))
            seconds=time.perf_counter()-t;train_seconds+=seconds
            next_lr=float(model.scheduler.step())
            vy,vp,vs=evaluate(model,data,'valid');val_seconds+=vs
            vm=metrics(vy,vp)['MSE']
            row=dict(epoch=epoch,lr_used=lr,lr_next=next_lr,validation_MSE=vm,training_seconds=seconds,validation_seconds=vs)
            row.update({k:float(np.average([r[k] for r in losses],weights=weights)) for k in losses[0]})
            history.append(row)
            if vm<best:
                best=vm;best_epoch=epoch
                torch.save(dict(states=model.states(),epoch=epoch,validation_MSE=vm),dest/'best_checkpoint.pt')
                np.savez(dest/'best_validation_predictions.npz',y_true=vy,y_pred=vp)
            write_csv(dest/'loss_history.csv',history)
            if epoch==1 or epoch%10==0:
                print(f'{name} epoch {epoch}/200 valid_MSE={vm:.8g} best={best:.8g}@{best_epoch} elapsed={time.perf_counter()-arm_start:.1f}s',flush=True)
        torch.save(dict(states=model.states(),epoch=200),dest/'final_checkpoint.pt')
        final_vm=history[-1]['validation_MSE']
        load(model,dest/'best_checkpoint.pt')
        vy,vp,_=evaluate(model,data,'valid');saved=np.load(dest/'best_validation_predictions.npz')
        assert np.array_equal(vy,saved['y_true']) and np.array_equal(vp,saved['y_pred'])
        # First pass includes matrix cache misses; second measures warmed pipeline.
        y,p,infer_cold=evaluate(model,data,'test')
        y2,p2,infer_warm=evaluate(model,data,'test')
        assert np.array_equal(y,y2) and np.array_equal(p,p2)
        test_keys=data.pairs['test'];ids=np.array([data.ids[b] for b,i in test_keys])
        np.savez(dest/'test_predictions.npz',y_true=y,y_pred=p,battery_id=ids,row_index=test_keys[:,1])
        result=dict(arm=name,seed=42,**metrics(y,p),training_runtime_seconds=train_seconds,
            validation_runtime_seconds=val_seconds,inference_cold_seconds=infer_cold,
            inference_warm_seconds=infer_warm,total_arm_seconds=time.perf_counter()-arm_start,
            best_epoch=best_epoch,stopping_epoch=200,early_stopping=False,best_validation_MSE=best,
            final_validation_MSE=final_vm,parameters=sum(p.numel() for m in model.modules for p in m.parameters()),
            checkpoint_verified=True)
        h.save_json(dest/'metrics.json',result);results.append(result);predictions[name]=(y,p,ids,test_keys[:,1])
        for b in np.unique(ids):
            mask=ids==b
            per_battery.append(dict(arm=name,seed=42,battery_id=b,**metrics(y[mask],p[mask])))
        print(name+' completed: '+json.dumps(result),flush=True)
    assert all(np.array_equal(predictions['W-O'][i],predictions['W-H'][i]) for i in (0,2,3))
    improvement={k:100*(results[0][k]-results[1][k])/results[0][k] if results[0][k] else None for k in ('MAE','MAPE_percent','RMSE','MSE')}
    for r in results:
        for k,v in improvement.items():r[k+'_improvement_percent']=v if r['arm']=='W-H' else None
    write_csv(out/'comparison_metrics.csv',results)
    originals={r['battery_id']:r for r in per_battery if r['arm']=='W-O'}
    for r in per_battery:
        for k in improvement:
            e=originals[r['battery_id']][k]
            r[k+'_improvement_percent']=100*(e-r[k])/e if r['arm']=='W-H' and e else None
    write_csv(out/'per_battery_metrics.csv',per_battery)
    h.save_json(out/'wang_improvement.json',improvement)
    plot_and_report(out,results,predictions,per_battery,improvement)


def plot_and_report(out,results,preds,per_battery,improvement):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(13,3.7))
    for ax,k in zip(axs,improvement):
        ax.bar(['Original','Matrix'],[r[k] for r in results],color=['#296a9c','#d17928'])
        ax.set_title(f'{k}\nImprovement {improvement[k]:+.2f}%');ax.set_ylabel('SOH fraction' if k in ('MAE','RMSE') else ('%' if k=='MAPE_percent' else 'SOH fraction squared'))
    fig.suptitle('MIT / HD-SOH-v1 / seed 42 / paired test samples');fig.tight_layout();fig.savefig(out/'overall_comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,name in zip(axs,('W-O','W-H')):
        y,p,ids,cycles=preds[name];ax.scatter(y,p,s=2,alpha=.2,rasterized=True)
        low=min(y.min(),p.min());high=max(y.max(),p.max());ax.plot([low,high],[low,high],'k--')
        ax.set(title=name,xlabel='True SOH',ylabel='Predicted SOH')
    fig.tight_layout();fig.savefig(out/'prediction_vs_true.png',dpi=150);plt.close(fig)
    folder=out/'prediction_vs_true';folder.mkdir(exist_ok=False)
    y,_,ids,cycles=preds['W-O']
    for b in np.unique(ids):
        mask=ids==b;fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(cycles[mask],y[mask],label='True',c='black')
        for name in ('W-O','W-H'):ax.plot(cycles[mask],preds[name][1][mask],label=name,alpha=.8)
        ax.set(title=b,xlabel='Original CSV row index',ylabel='SOH fraction');ax.legend();fig.tight_layout();fig.savefig(folder/(b+'.png'),dpi=130);plt.close(fig)
    lines=['# Wang MIT 公平比較結果','',
        '只完成 W-O / W-H，各seed42一次；Zhang沒有執行。四組預檢的smoke numbers沒有用作正式結果。','',
        '共同protocol：HD-SOH-v1；71/18/36 train/validation/test batteries；42,784/13,585/29,083 pairs；SOH=capacity/1.1；train-only min-max，batch512，各200 epochs，不early-stop。',
        '模型input與loss均沿用計畫及預檢；沒有改原模型檔案。W-H第一層較寬並使用16-cycle history，因此本結果是pipeline比較，不能單獨歸因於FFT matrix。','',
        '| Metric | Wang original | Wang matrix | Improvement % |','| --- | ---: | ---: | ---: |']
    for k,v in improvement.items():lines.append(f'| {k} | {results[0][k]:.9g} | {results[1][k]:.9g} | {v:+.3f} |')
    lines+=['','Improvement = (original error - matrix error) / original error × 100；負值表示matrix較差。MAPE以真值為分母，單位%；MAE/RMSE是SOH fraction，MSE是其平方。','',
        '| Arm | Training s | Validation s | Inference cold / warm s | Best epoch | End epoch |',
        '| --- | ---: | ---: | --- | ---: | ---: |']
    for r in results:lines.append(f"| {r['arm']} | {r['training_runtime_seconds']:.2f} | {r['validation_runtime_seconds']:.2f} | {r['inference_cold_seconds']:.2f} / {r['inference_warm_seconds']:.2f} | {r['best_epoch']} | 200 |")
    lines+=['','Training runtime包含batch準備及matrix cache misses；validation另外列出。Inference是完整test資料pipeline耗時，不是純network kernel時間。冷／暖cache是同一checkpoint，已確認predictions完全相同。','',
        '每個模型按自己的validation MSE選best checkpoint，reload後validation predictions逐元素完全一致。Test sample keys／labels已逐元素確認一致。Initial seed、source snapshots、split、scaler、hashes和設定在wang_shared/；每arm checkpoint、loss_history、predictions、metrics在各seed_42/。','',
        'comparison_metrics.csv是pooled sample-weighted metrics；per_battery_metrics.csv含36顆電池各兩組的metrics及paired improvement。未把cycles當獨立統計replicates；單seed不代表普遍優勝或顯著性。','',
        '輸出包括overall_comparison.png、prediction_vs_true.png及36張per-battery SOH trajectory plots。未啟動Zhang或其他seed。']
    (out/'ANALYSIS_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
