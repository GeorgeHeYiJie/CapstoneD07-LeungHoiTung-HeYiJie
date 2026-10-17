"""Run only Zhang SOH adaptation Z-O and Z-H, seed 42, under frozen HD-SOH-v1."""
import argparse
import hashlib
import platform
import subprocess
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
    rng=np.random.default_rng(20042 if split=='valid' else 30042)
    for i in range(0,len(keys),h.BATCH):
        y,p=model.predict(data,keys[i:i+h.BATCH],rng);ys.append(y);ps.append(p)
    return np.concatenate(ys),np.concatenate(ps),time.perf_counter()-start


def load(model,path):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    for m,state in zip(model.modules,obj['states']):m.load_state_dict(state)
    return obj


def main():
    seed=42
    h.SEED=seed
    torch.set_num_threads(8);torch.set_num_interop_threads(8)
    out=h.OUT;shared=out/'zhang_shared_seed_42';shared.mkdir(exist_ok=False)
    shutil.copy2(out/'ANALYSIS_SUMMARY.md',shared/'previous_analysis_summary.md')
    start=time.perf_counter();data=h.Data();data.write_records(shared)
    # Exact match with the previously measured and accepted protocol.
    for filename in ('split_manifest.json','scaler.json','dataset_manifest.json'):
        assert json.loads((shared/filename).read_text())==json.loads((out/'preflight'/filename).read_text())
    h.save_json(shared/'checks.json',data.checks())
    prep=time.perf_counter()-start
    for name in ('run_zhang_comparison.py','high_dimension_comparison.py'):
        shutil.copy2(Path(__file__).parent/name,shared/name)
    h.save_json(shared/'configuration.json',dict(protocol='HD-SOH-v1',arms=['Z-O','Z-H'],seed=seed,
        split_seed=420,batch_size=512,epochs=200,early_stopping=False,threads=8,
        checkpoint='each arm own lowest validation MSE',normalization='frozen train-only minmax',
        alpha_mix=.5,lambda_delta=1.,learning_rate=.001,scheduler=None,
        training_reference_seed=10042,validation_reference_seed=20042,test_reference_seed=30042,evaluation_references=2,
        shared_data_load_seconds=prep,environment=json.loads((out/'preflight/config.json').read_text())))
    h.save_json(shared/'provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(h.__file__),h.ROOT/'Model/Model.py',h.ROOT/'HighDimensionMatrix-PINN_Combination.py']},seed=seed,note='Existing ZhangSOH classes and HD-SOH-v1 unchanged; split 420; seed 42'))
    shutil.copy2(out/'final_comparison.csv',shared/'previous_final_comparison.csv')
    reference_checks={}
    for split,refseed in [('valid',20042),('test',30042)]:
        keys=data.pairs[split];refs=data.references(keys,np.random.default_rng(refseed),2)
        assert np.isin(refs,data.train_pool).all() and np.all(refs[:,0]!=refs[:,1])
        rows=np.minimum(keys[:,1,None],np.array([len(data.labels[b])-1 for b in refs.flat]).reshape(refs.shape))
        np.savez(shared/(split+'_references.npz'),target_battery_id=np.array(data.ids)[keys[:,0]],target_row=keys[:,1],reference_battery_id=np.array(data.ids)[refs],reference_row=rows)
        reference_checks[split]=dict(references=len(refs)*2,clamped=int((rows!=keys[:,1,None]).sum()),training_only=True,hash=h.digest(refs))
    h.save_json(shared/'reference_checks.json',reference_checks)
    # Bounded fresh one-batch update + checkpoint roundtrip before long runs.
    checks={}
    for name in ('Z-O','Z-H'):
        m=h.Arm(name);keys=data.pairs['train'][:h.BATCH]
        before=next(m.modules[0].parameters()).detach().clone()
        loss=m.step(data,keys,np.random.default_rng(10000+seed))
        assert not torch.equal(before,next(m.modules[0].parameters()))
        y,p=m.predict(data,keys,np.random.default_rng(40042))
        cp=shared/(name+'_diagnostic.pt');torch.save(dict(states=m.states()),cp)
        load(m,cp);_,p2=m.predict(data,keys,np.random.default_rng(40042))
        assert np.array_equal(p,p2)
        for split in ['valid','test']:
            sy,sp=m.predict(data,data.pairs[split][:h.BATCH],np.random.default_rng(50042))
            assert sy.shape==sp.shape and np.isfinite(sp).all()
        _,rep,target=data.batch(keys,name)
        checks[name]=dict(one_batch_loss=loss,input_shape=list(rep.shape),target_shape=list(target.shape),parameters_updated=True,checkpoint_exact=True,validation_test_smoke_passed=True)
    h.save_json(shared/'runtime_checks.json',checks)
    results=[];predictions={};per_battery=[]
    for name in ('Z-O','Z-H'):
        dest=out/name/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=False)
        model=h.Arm(name) # Fresh seed; never continues diagnostic weights.
        torch.save(dict(torch_rng=torch.get_rng_state(),states=model.states()),dest/'initial_state.pt')
        order_rng=np.random.default_rng(seed);ref_rng=np.random.default_rng(10000+seed)
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
            next_lr=lr # Zhang uses constant Adam learning rate; no scheduler.
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
        wang=np.load(out/'W-O/seed_42/test_predictions.npz')
        assert np.array_equal(y,wang['y_true']) and np.array_equal(ids,wang['battery_id']) and np.array_equal(test_keys[:,1],wang['row_index'])
        result=dict(arm=name,seed=seed,**metrics(y,p),training_runtime_seconds=train_seconds,
            validation_runtime_seconds=val_seconds,inference_cold_seconds=infer_cold,
            inference_warm_seconds=infer_warm,total_arm_seconds=time.perf_counter()-arm_start,
            best_epoch=best_epoch,stopping_epoch=200,early_stopping=False,best_validation_MSE=best,
            final_validation_MSE=final_vm,parameters=sum(p.numel() for m in model.modules for p in m.parameters()),
            checkpoint_verified=True)
        h.save_json(dest/'metrics.json',result);results.append(result);predictions[name]=(y,p,ids,test_keys[:,1])
        for b in np.unique(ids):
            mask=ids==b
            per_battery.append(dict(arm=name,seed=seed,battery_id=b,**metrics(y[mask],p[mask])))
        print(name+' completed: '+json.dumps(result),flush=True)
    assert all(np.array_equal(predictions['Z-O'][i],predictions['Z-H'][i]) for i in (0,2,3))
    write_csv(shared/'run_metrics.csv',results)
    write_csv(shared/'per_battery_metrics.csv',per_battery)

if __name__=='__main__':main()
