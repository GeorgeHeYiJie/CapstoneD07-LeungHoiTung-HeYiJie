"""Offline denoising experiment adapters. Source algorithms remain unchanged.

prepare -> smoke (all four conditions) -> run -> analyze.
Never use smoke scores as formal results. See results/denoising_analysis/PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import high_dimension_comparison as h
from run_wang_comparison import metrics, write_csv

ROOT = h.ROOT
OUT = ROOT / 'results/denoising_analysis'
PREP = OUT / 'prepared'
SOURCES = ['DenoisingAlgorithm.py','HighDimensionMatrix-PINN_Combination.py',
           'New1HighDimensionMatrix-PINN_Combination.py']
spec = importlib.util.spec_from_file_location('new1_source', ROOT/SOURCES[2])
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
hd = h.hd
EPOCHS = 200
SEED = 42
LR_ARGS = SimpleNamespace(epochs=200,warmup_epochs=30,warmup_lr=.002,lr=.01,final_lr=.0002)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    os.replace(temp,path)


def denoiser():
    # Retain original imports/constants/functions; omit ONLY module-level mkdir
    # and the main guard. No rewrite of filtering functions or numerical constants.
    tree=ast.parse((ROOT/SOURCES[0]).read_text(encoding='utf8'))
    tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef))]
    ns={'__name__':'denoising_offline_adapter'}
    exec(compile(tree,str(ROOT/SOURCES[0]),'exec'),ns)
    return ns


def prepare():
    PREP.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); base=h.Data();base.write_records(PREP/'original' if False else PREP)
    frozen=ROOT/'results/high_dimension_comparison/wang_shared'
    for name in ['split_manifest.json','dataset_manifest.json','scaler.json']:
        assert json.loads((PREP/name).read_text())==json.loads((frozen/name).read_text()),name
    original_keys=base.pairs
    ns=denoiser(); paths=sorted((ROOT/'data/MIT data').glob('*/*.csv'),key=lambda p:p.stem)
    for path in paths:
        target=PREP/'denoised_csv'/path.parent.name
        target.mkdir(parents=True,exist_ok=True)
        ns['OUTPUT_DIR']=str(target)
        ns['process_one_file'](str(path))
    raw_d=[]; manifest=[]
    for path in paths:
        a=pd.read_csv(path); dest=PREP/'denoised_csv'/path.parent.name/path.name
        b=pd.read_csv(dest,float_precision='round_trip')
        assert list(a.columns)==list(b.columns) and len(a)==len(b)
        assert np.array_equal(a.capacity.to_numpy(),b.capacity.to_numpy()),path.name
        raw_d.append(b[h.hd.FEATURE_COLUMNS].to_numpy(dtype=np.float64))
        assert np.isfinite(raw_d[-1]).all()
        manifest.append(dict(battery=path.stem,rows=len(a),source_sha256=sha(path),denoised_sha256=sha(dest),capacity_exact=True))
    offsets=np.cumsum([0]+[len(x) for x in base.raw])
    labels=np.concatenate(base.labels)
    np.save(PREP/'labels.npy',labels)
    np.save(PREP/'offsets.npy',offsets)
    for split,keys in original_keys.items():np.save(PREP/(split+'_keys.npy'),keys)
    # Original Test 2 training loop: each non-overlapping adjacent pair, stride 2.
    keys2=original_keys['train'][original_keys['train'][:,1]%2==0]
    np.save(PREP/'test2_train_keys.npy',keys2)
    record=dict(ids=base.ids,splits=base.splits,rows=[len(x) for x in base.labels],
                split_counts={k:len(v) for k,v in base.splits.items()},
                pair_counts={k:len(v) for k,v in original_keys.items()},
                key_hashes={k:h.digest(v) for k,v in original_keys.items()},
                label_hash=h.digest(labels),test2_training_updates_per_epoch=len(keys2),
                test2_training_key_hash=h.digest(keys2),test_samples=29083)
    refcheck={}
    for split,seed in [('valid',20042),('test',30042)]:
        refs=base.references(original_keys[split],np.random.default_rng(seed),2)
        rows=np.minimum(original_keys[split][:,1,None],np.array(record['rows'])[refs]-1)
        flat=offsets[refs]+rows
        assert np.isin(refs,base.train_pool).all()
        np.save(PREP/(split+'_references.npy'),flat)
        refcheck[split]=dict(seed=seed,hash=h.digest(flat),train_only=True)
    for condition,raw in [('original',base.raw),('denoised',raw_d)]:
        folder=PREP/condition;folder.mkdir()
        fit=np.concatenate([raw[i] for i in base.splits['train']])
        low=fit.min(0);high=fit.max(0);span=np.where(high==low,1.,high-low);med=np.median(fit,0)
        feats=[h.normalize_features(a,low,span,med) for a in raw]
        xt=[np.column_stack([x,2*np.arange(len(x))/base.max_cycle-1]).astype(np.float32) for x in feats]
        np.save(folder/'features.npy',np.concatenate(feats));np.save(folder/'xt.npy',np.concatenate(xt))
        save(folder/'scaler.json',dict(low=low.tolist(),span=span.tolist(),median=med.tolist(),max_train_cycle=base.max_cycle,fit_batteries=[base.ids[b] for b in base.splits['train']]))
    prior=np.load(ROOT/'results/high_dimension_comparison/W-O/seed_42/test_predictions.npz')
    tk=original_keys['test']; ids=np.array(base.ids)[tk[:,0]]
    assert np.array_equal(ids,prior['battery_id']) and np.array_equal(tk[:,1],prior['row_index'])
    assert np.array_equal(labels[offsets[tk[:,0]]+tk[:,1]].reshape(-1,1),prior['y_true'])
    # Empirical causality check: alter future features in a train battery only.
    signal=base.raw[base.splits['train'][0]][:,0].copy()
    def filt(s):
        first=ns['sg_gcl_denoise'](s,ns['SG_WINDOW'],ns['SG_POLY'],ns['GCL_SIGMA'],ns['GCL_ALPHA'],ns['GCL_ITER'])
        return ns['multi_scale_decompose_and_fuse'](first,ns['SCALES'],ns['FUSION_MODE'],ns['SCALE_WEIGHTS'])
    changed=signal.copy();changed[21:]+=1.
    future_delta=float(np.max(np.abs(filt(signal)[:21]-filt(changed)[:21])))
    save(PREP/'manifest.json',record)
    save(PREP/'verification.json',dict(source_csvs=manifest,original_denoised_ids_rows_labels_exact=True,
        previous_test_ids_rows_labels_exact=True,no_outlier_filter=True,train_only_scalers=True,
        references=refcheck,offline_future_perturbation_max_past_change=future_delta,
        preparation_seconds=time.perf_counter()-start))
    snapshot=PREP/'source';snapshot.mkdir()
    for name in SOURCES:shutil.copy2(ROOT/name,snapshot/name)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    save(PREP/'provenance.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        source_hashes={name:sha(ROOT/name) for name in SOURCES},seed=42,offline=True))
    print('PREPARED: all frozen samples/labels match; future-information use documented.',flush=True)


class Data:
    def __init__(self,condition,test,build=True):
        self.condition=condition;self.test=test
        self.manifest=json.loads((PREP/'manifest.json').read_text())
        self.ids=self.manifest['ids'];self.splits=self.manifest['splits']
        self.offsets=np.load(PREP/'offsets.npy');self.y=torch.from_numpy(np.load(PREP/'labels.npy')).reshape(-1,1)
        self.features=torch.from_numpy(np.load(PREP/condition/'features.npy'))
        self.xt=torch.from_numpy(np.load(PREP/condition/'xt.npy'))
        self.keys={s:np.load(PREP/(s+'_keys.npy')) for s in ['train','valid','test']}
        self.trainkeys=self.keys['train'] if test=='test1' else np.load(PREP/'test2_train_keys.npy')
        self.pool=np.array(self.splits['train'])
        self.refs={s:np.load(PREP/(s+'_references.npy')) for s in ['valid','test']}
        self.matrix_path=PREP/condition/(test+'_matrices.npy')
        if build:self.materialize()

    def materialize(self):
        if not self.matrix_path.exists():
            start=time.perf_counter(); k=2 if self.test=='test1' else 3
            tmp=self.matrix_path.with_suffix('.building.npy')
            mm=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(self.y),k,16,8,8))
            for b in range(len(self.ids)):
                a,z=self.offsets[b:b+2];feat=self.features[a:z].numpy()
                for i in range(len(feat)):
                    mm[a+i]=hd.matrices_from_window(hd.causal_window(feat,i,16),2,8,8) if self.test=='test1' else new.build_matrix(feat,i,3,8,8)
                if b%25==0:print(f'{self.test}/{self.condition} matrix cache {b+1}/125',flush=True)
            mm.flush();del mm;os.replace(tmp,self.matrix_path)
            save(self.matrix_path.with_suffix('.json'),dict(seconds=time.perf_counter()-start,shape=[len(self.y),k,16,8,8],source_function='hd.matrices_from_window' if k==2 else 'new.build_matrix'))
        self.matrix=np.load(self.matrix_path,mmap_mode='r')

    def flat(self,keys,second=0):return self.offsets[keys[:,0]]+keys[:,1]+second

    def mat(self,flat):return torch.from_numpy(np.array(self.matrix[flat],copy=True))

    def batch(self,keys,second=0):
        flat=self.flat(keys,second);ref0=self.offsets[keys[:,0]]
        mat=self.mat(flat)
        return self.xt[flat],mat,self.y[flat],mat-self.mat(ref0)

    def reference(self,keys,rng):
        ref=np.array([rng.choice(self.pool[self.pool!=b]) for b,_ in keys])
        rows=np.minimum(keys[:,1],np.array(self.manifest['rows'])[ref]-1)
        rows2=np.minimum(keys[:,1]+1,np.array(self.manifest['rows'])[ref]-1)
        a=self.offsets[ref]+rows;c=self.offsets[ref]+rows2;base=self.mat(self.offsets[ref])
        return self.mat(a)-base,self.mat(c)-base,self.y[a],self.y[c]


class Test1:
    names=('wang','zhang')
    batch=64
    def __init__(self):
        h.seed_all()
        self.u=hd.SolutionU(2048);self.f=hd.MLP(35,1,3,60,.2);self.z=hd.ZhangSOH(2,16)
        self.modules=[self.u,self.f,self.z]
        self.opts=[torch.optim.Adam(self.u.parameters(),lr=.002),torch.optim.Adam(self.f.parameters(),lr=.001),torch.optim.Adam(self.z.parameters(),lr=.001)]

    def step(self,d,keys,rng):
        for m in self.modules:m.train()
        for o in self.opts:o.zero_grad(set_to_none=True)
        x1,m1,y1,i1=d.batch(keys);x2,m2,y2,i2=d.batch(keys,1)
        r1,r2,yr1,yr2=d.reference(keys,rng)
        u1,f1=hd.wang_forward(self.u,self.f,x1,m1);u2,f2=hd.wang_forward(self.u,self.f,x2,m2)
        mse=torch.nn.functional.mse_loss
        a=.5*mse(u1,y1)+.5*mse(u2,y2);b=.5*f1.square().mean()+.5*f2.square().mean()
        c=torch.relu((u2-u1)*(y1-y2)).sum();w=a+.5*b+.01*c
        intra=.5*mse(self.z.intra(i1),y1)+.5*mse(self.z.intra(i2),y2)
        delta=.5*mse(self.z.delta(i1,r1),y1-yr1)+.5*mse(self.z.delta(i2,r2),y2-yr2);z=intra+delta
        if not torch.isfinite(w+z):raise RuntimeError('Nonfinite training loss')
        w.backward();z.backward()
        for module in self.modules:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        for o in self.opts:o.step()
        return dict(data=a.item(),pde=b.item(),physics=c.item(),wang_total=w.item(),intra=intra.item(),delta=delta.item(),zhang_total=z.item())

    @torch.no_grad()
    def predict(self,d,keys,refs):
        for m in self.modules:m.eval()
        xt,mat,y,intra=d.batch(keys)
        wang=hd.predict_wang(self.u,xt,mat)
        parts=[];base_indices=np.searchsorted(d.offsets[1:],refs,side='right')
        for j in range(2):
            r=refs[:,j];rep=d.mat(r)-d.mat(d.offsets[base_indices[:,j]])
            parts.append(self.z.mixture(self.z.intra(intra),self.z.delta(intra,rep),d.y[r],.5))
        return y.numpy(),dict(wang=wang.numpy(),zhang=torch.stack(parts).mean(0).numpy())

    def states(self,name):return [self.u.state_dict(),self.f.state_dict()] if name=='wang' else [self.z.state_dict()]
    def load(self,name,states):
        for m,s in zip([self.u,self.f] if name=='wang' else [self.z],states):m.load_state_dict(s)


class Test2:
    names=('combined',)
    batch=1  # one key = two successive rows in the source training loop
    def __init__(self):
        h.seed_all();self.model=new.CombinedModel();self.modules=[self.model]
        self.opts=[torch.optim.Adam(self.model.parameters(),lr=.001)]

    def step(self,d,keys,rng):
        self.model.train();flat=d.flat(keys)[0];idx=np.array([flat,flat+1])
        # Sole runtime repair: enable differentiation w.r.t. the original xt.
        xt=d.xt[idx].clone().requires_grad_(True);mat=d.mat(idx);y=d.y[idx];feat=d.features[idx]
        pred,u=self.model(xt,mat,feat);f=self.model.wang.physics_residual(xt,u,mat)
        a=torch.nn.functional.mse_loss(pred,y);b=f.square().mean()
        c=torch.relu((u[1]-u[0])*(y[0]-y[1])).sum()
        loss=a+new.ALPHA_PHYS*b+new.BETA_MONO*c
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        self.opts[0].zero_grad(set_to_none=True);loss.backward()
        for n,p in self.model.named_parameters():
            if n.startswith('zhang.head_delta.'):assert p.grad is None # source never uses this head
            else:assert p.grad is not None and torch.isfinite(p.grad).all(),n
        self.opts[0].step()
        return dict(data=a.item(),pde=b.item(),physics=c.item(),total=loss.item())

    @torch.no_grad()
    def predict(self,d,keys,refs=None):
        self.model.eval();flat=d.flat(keys)
        pred,_=self.model(d.xt[flat],d.mat(flat),d.features[flat])
        return d.y[flat].numpy(),dict(combined=pred.numpy())

    def states(self,name):return [self.model.state_dict()]
    def load(self,name,states):self.model.load_state_dict(states[0])


def evaluate(model,d,split,limit=None):
    keys=d.keys[split] if limit is None else d.keys[split][:limit]
    ys=[];ps={n:[] for n in model.names};start=time.perf_counter()
    for i in range(0,len(keys),256):
        y,p=model.predict(d,keys[i:i+256],d.refs[split][i:i+256]);ys.append(y)
        for n in ps:
            assert np.isfinite(p[n]).all() and p[n].shape==y.shape
            ps[n].append(p[n])
    return np.concatenate(ys),{n:np.concatenate(v) for n,v in ps.items()},time.perf_counter()-start


def configuration(test,condition):
    return dict(test=test,condition=condition,seed=42,epochs=200,early_stopping=False,offline=True,
        checkpoint_rule='each predictor independently: minimum full frozen validation SOH MSE',
        normalization='condition-specific training-only minmax; same protocol; no outlier filtering',
        eval_samples=29083,threads=1,device='cpu',
        batch_size=64 if test=='test1' else 2,training_stride=1 if test=='test1' else 2,
        matrix_periods=2 if test=='test1' else 3,window=16,height=8,width=8,
        optimizer='Adam defaults',lr={'warmup':.002,'base':.01,'final':.0002,'dynamics':.001,'zhang':.001,'warmup_epochs':30} if test=='test1' else .001,
        alpha=.5 if test=='test1' else .1,beta=.01,
        validation_reference_seed=20042,test_reference_seed=30042,training_reference_seed=10042,
        source_hashes={name:sha(ROOT/name) for name in SOURCES},runner_hash=sha(__file__))


def smoke():
    folder=OUT/'smoke';folder.mkdir(exist_ok=False)
    rows=[]
    for test,cls in [('test1',Test1),('test2',Test2)]:
        initial=None
        for condition in ['original','denoised']:
            d=Data(condition,test);m=cls();state=copy.deepcopy([x.state_dict() for x in m.modules])
            if initial is not None:
                assert all(torch.equal(initial[i][k],v) for i,s in enumerate(state) for k,v in s.items())
            initial=state
            rng=np.random.default_rng(10042);times=[];losses=[]
            n=20 if test=='test1' else 100
            before=next(m.modules[0].parameters()).detach().clone()
            for j in range(n):
                batch=d.trainkeys[j*m.batch:(j+1)*m.batch]
                t=time.perf_counter();losses.append(m.step(d,batch,rng));times.append(time.perf_counter()-t)
            assert not torch.equal(before,next(m.modules[0].parameters()))
            vy,vp,vt=evaluate(m,d,'valid',512)
            # Smoke uses training/validation only; reserved test targets are not scored.
            for name in m.names:
                cp=folder/f'{test}_{condition}_{name}.pt';torch.save(m.states(name),cp)
                m.load(name,torch.load(cp,weights_only=True));_,p,_=evaluate(m,d,'valid',512)
                assert np.array_equal(vp[name],p[name])
            steps=int(np.ceil(len(d.trainkeys)/m.batch));seconds=float(np.median(times[5:]))
            estimate=200*(steps*seconds+vt*len(d.keys['valid'])/512)
            row=dict(test=test,condition=condition,updates=n,finite=True,checkpoint_roundtrip=True,
                same_pair_initial_weights=True,median_update_seconds=seconds,updates_per_epoch=steps,
                estimated_total_hours=estimate/3600,estimated_with_30pct_margin_hours=estimate*1.3/3600,
                last_diagnostic_loss=losses[-1],matrix_shape=list(d.matrix.shape),configuration=configuration(test,condition))
            rows.append(row);save(folder/'checks.json',dict(complete=False,conditions=rows))
            print('SMOKE '+json.dumps(row),flush=True)
            del d,m
    save(folder/'checks.json',dict(complete=True,conditions=rows,total_estimated_hours=sum(r['estimated_total_hours'] for r in rows)))


def run(test,condition):
    assert json.loads((OUT/'smoke/checks.json').read_text())['complete']
    expected=json.loads((PREP/'provenance.json').read_text())['source_hashes']
    assert all(sha(ROOT/n)==v for n,v in expected.items())
    dest=OUT/test/condition;dest.mkdir(parents=True,exist_ok=False)
    save(dest/'configuration.json',configuration(test,condition))
    shutil.copy2(__file__,dest/'runner_snapshot.py')
    d=Data(condition,test);m=Test1() if test=='test1' else Test2()
    torch.save({n:m.states(n) for n in m.names},dest/'initial_state.pt')
    best={n:float('inf') for n in m.names};best_epoch={n:None for n in m.names}
    history=[];order_rng=np.random.default_rng(42);ref_rng=np.random.default_rng(10042)
    start=time.perf_counter();training=validation=0.
    for ep in range(1,201):
        keys=d.trainkeys
        if test=='test1':
            keys=keys[order_rng.permutation(len(keys))]
            m.opts[0].param_groups[0]['lr']=float(hd.solution_lr(ep-1,LR_ARGS))
        sums={};weight=0;t=time.perf_counter()
        for j in range(0,len(keys),m.batch):
            batch=keys[j:j+m.batch];parts=m.step(d,batch,ref_rng)
            for k,v in parts.items():sums[k]=sums.get(k,0.)+v*len(batch)
            weight+=len(batch)
            if j and j%(m.batch*1000)==0:
                save(dest/'progress.json',dict(epoch=ep,epochs=200,updates=j//m.batch,updates_total=int(np.ceil(len(keys)/m.batch)),elapsed_seconds=time.perf_counter()-start))
        ts=time.perf_counter()-t;training+=ts
        y,p,vs=evaluate(m,d,'valid');validation+=vs
        row=dict(epoch=ep,training_seconds=ts,validation_seconds=vs,lr=m.opts[0].param_groups[0]['lr'],**{k:v/weight for k,v in sums.items()})
        for n in m.names:
            vm=metrics(y,p[n])['MSE'];row[n+'_validation_MSE']=vm
            if vm<best[n]:
                best[n]=vm;best_epoch[n]=ep
                torch.save(dict(states=m.states(n),epoch=ep,validation_MSE=vm),dest/(n+'_best.pt'))
                np.savez_compressed(dest/(n+'_best_validation.npz'),y_true=y,y_pred=p[n])
        history.append(row);write_csv(dest/'loss_history.csv',history)
        save(dest/'progress.json',dict(epoch=ep,epochs=200,best_epoch=best_epoch,best_validation_MSE=best,elapsed_seconds=time.perf_counter()-start,complete=False))
        print(f'{test}/{condition} epoch {ep}/200 best={best_epoch} valid={ {n:row[n+"_validation_MSE"] for n in m.names} } elapsed={time.perf_counter()-start:.1f}s',flush=True)
    torch.save({n:m.states(n) for n in m.names},dest/'final_state.pt')
    for n in m.names:m.load(n,torch.load(dest/(n+'_best.pt'),weights_only=True)['states'])
    vy,vp,_=evaluate(m,d,'valid')
    for n in m.names:
        prev=np.load(dest/(n+'_best_validation.npz'))
        assert np.array_equal(prev['y_true'],vy) and np.array_equal(prev['y_pred'],vp[n])
    y,p,infer=evaluate(m,d,'test')
    keys=d.keys['test'];ids=np.array(d.ids)[keys[:,0]];rows=keys[:,1]
    old=np.load(ROOT/'results/high_dimension_comparison/W-O/seed_42/test_predictions.npz')
    assert np.array_equal(y,old['y_true']) and np.array_equal(ids,old['battery_id']) and np.array_equal(rows,old['row_index'])
    results=[]
    for n in m.names:
        np.savez_compressed(dest/(n+'_test_predictions.npz'),y_true=y,y_pred=p[n],battery_id=ids,row_index=rows,
                            trajectory_length=np.array(d.manifest['rows'])[keys[:,0]])
        result=dict(test=test,condition=condition,predictor=n,seed=42,**metrics(y,p[n]),best_epoch=best_epoch[n],
            stopping_epoch=200,best_validation_MSE=best[n],final_validation_MSE=history[-1][n+'_validation_MSE'],
            checkpoint=n+'_best.pt',checkpoint_verified=True,training_seconds=training,validation_seconds=validation,
            test_inference_seconds=infer,total_run_seconds=time.perf_counter()-start,offline=True)
        results.append(result)
    save(dest/'metrics.json',results)
    save(dest/'progress.json',dict(complete=True,epoch=200,best_epoch=best_epoch,total_seconds=time.perf_counter()-start))
    print('COMPLETE '+json.dumps(results),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','smoke','run'])
    parser.add_argument('--test',choices=['test1','test2']);parser.add_argument('--condition',choices=['original','denoised'])
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.action=='prepare':prepare()
    elif args.action=='smoke':smoke()
    else:
        if not args.test or not args.condition:parser.error('run requires --test and --condition')
        run(args.test,args.condition)

if __name__=='__main__':main()
