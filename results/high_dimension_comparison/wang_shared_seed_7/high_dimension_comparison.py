"""HD-SOH-v1: isolated four-arm preflight and runtime estimator.

Imports existing model classes without editing them. New data protocol and
input shapes are those specified in docs/HIGH_DIMENSION_ANALYSIS_PLAN.md.
Never uses archived smoke predictions. This script does not launch full runs.
The user requires a stop if total estimated training exceeds one hour.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from Model.Model import Solution_u, MLP, LR_Scheduler

spec = importlib.util.spec_from_file_location('hd_source', ROOT / 'HighDimensionMatrix-PINN_Combination.py')
hd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd)
ARMS = ('W-O', 'W-H', 'Z-O', 'Z-H')
OUT = ROOT / 'results/high_dimension_comparison'
BATCH = 512
EPOCHS = 200
SEED = 42


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def normalize_features(raw, low, span, med):
    # Each column uses past-only forward fill, then a training-only fallback.
    frame = pd.DataFrame(raw).replace([np.inf, -np.inf], np.nan).ffill()
    filled = frame.fillna(pd.Series(med)).to_numpy(dtype=np.float64)
    return (2 * (filled - low) / span - 1).astype(np.float32)


def window_at(features, i):
    start = max(0, i - 15)
    w = features[start:i + 1]
    if len(w) < 16:
        w = np.concatenate([np.repeat(features[:1], 16 - len(w), axis=0), w])
    return np.ascontiguousarray(w)


def period_matrix(w):
    # Explicitly exclude DC. Stable frequency-index tie break, per HD-SOH-v1.
    amp = np.abs(np.fft.rfft(w, axis=0)).mean(axis=1)
    freq = np.arange(1, len(amp))
    selected = freq[np.lexsort((freq, -amp[1:]))[:2]]
    mats = []
    s = torch.from_numpy(w)
    for f in selected:
        p = 16 // int(f)
        pad = (-16) % p
        v = torch.cat([s, s.new_zeros(pad, 16)]) if pad else s
        grid = v.reshape(-1, p, 16).permute(2, 0, 1).unsqueeze(0)
        mats.append(torch.nn.functional.interpolate(grid, (8, 8), mode='bilinear', align_corners=False)[0])
    return torch.stack(mats)


class Data:
    def __init__(self):
        paths = sorted((ROOT / 'data/MIT data').glob('*/*.csv'), key=lambda p: p.stem)
        assert len(paths) == 125
        self.ids = [p.stem for p in paths]
        test = [i for i, p in enumerate(paths) if p.parent.name == '2018-04-12']
        rest = [i for i in range(len(paths)) if i not in test]
        train, valid = train_test_split(rest, test_size=.2, random_state=420)
        self.splits = dict(train=list(train), valid=list(valid), test=test)
        assert [len(self.splits[k]) for k in self.splits] == [71, 18, 36]
        self.raw, self.labels = [], []
        for p in paths:
            df = pd.read_csv(p)
            self.raw.append(df[hd.FEATURE_COLUMNS].to_numpy(dtype=np.float64))
            self.labels.append((df['capacity'].to_numpy(dtype=np.float64) / 1.1).astype(np.float32))
        fit = np.concatenate([self.raw[i] for i in train])
        fit[~np.isfinite(fit)] = np.nan
        self.low, high, self.med = np.nanmin(fit, axis=0), np.nanmax(fit, axis=0), np.nanmedian(fit, axis=0)
        assert np.isfinite([self.low, high, self.med]).all()
        self.span = np.where(high == self.low, 1., high - self.low)
        self.max_cycle = max(len(self.raw[i]) - 1 for i in train)
        self.features = [normalize_features(a, self.low, self.span, self.med) for a in self.raw]
        self.xt = [torch.from_numpy(np.column_stack([a, 2*np.arange(len(a))/self.max_cycle-1]).astype(np.float32)) for a in self.features]
        self.pairs = {}
        for split, owners in self.splits.items():
            self.pairs[split] = np.array([(b, i) for b in owners for i in range(len(self.labels[b])-1)
                                          if np.isfinite(self.labels[b][i:i+2]).all()], dtype=np.int64)
        # Cached matrices are per cycle, not four duplicated pair tensors.
        self.cache = {}
        self.original_cache = {}
        self.train_pool = np.array(train)
        assert all(np.isfinite(a).all() for a in self.features)
        assert all(np.isfinite(self.labels[b]).all() for b in train), 'Reference labels require explicit missing-label policy'

    def representation(self, b, i, matrix):
        cache = self.cache if matrix else self.original_cache
        key = (int(b), int(i))
        if key not in cache:
            w = window_at(self.features[b], i)
            cache[key] = period_matrix(w) if matrix else torch.from_numpy(w.T.copy()).reshape(1, 16, 16, 1)
        return cache[key]

    def intra(self, b, i, matrix):
        return self.representation(b, i, matrix) - self.representation(b, 0, matrix)

    def batch(self, keys, arm, second=False):
        keys = [(int(b), int(i) + int(second)) for b, i in keys]
        y = torch.tensor([self.labels[b][i] for b, i in keys]).reshape(-1, 1)
        xt = torch.stack([self.xt[b][i] for b, i in keys])
        rep = None
        if arm == 'W-H':
            rep = torch.stack([self.representation(b, i, True) for b, i in keys])
        elif arm.startswith('Z'):
            rep = torch.stack([self.intra(b, i, arm == 'Z-H') for b, i in keys])
        return xt, rep, y

    def references(self, keys, rng, n):
        refs = []
        for b, i in keys:
            choices = self.train_pool[self.train_pool != b]
            refs.append(rng.choice(choices, size=n, replace=False))
        return np.asarray(refs)

    def ref_batch(self, keys, refs, arm, second=False):
        rep, labels = [], []
        for (_, i), b in zip(keys, refs):
            j = min(int(i) + int(second), len(self.labels[b])-1)
            rep.append(self.intra(int(b), j, arm == 'Z-H'))
            labels.append(self.labels[b][j])
        return torch.stack(rep), torch.tensor(labels).reshape(-1, 1)

    def write_records(self, dest):
        save_json(dest / 'split_manifest.json', {k:[self.ids[i] for i in v] for k,v in self.splits.items()})
        save_json(dest / 'scaler.json', dict(low=self.low.tolist(), span=self.span.tolist(), median=self.med.tolist(), max_train_cycle=self.max_cycle))
        save_json(dest / 'dataset_manifest.json', dict(
            pairs={k:len(v) for k,v in self.pairs.items()},
            key_hashes={k:digest(v) for k,v in self.pairs.items()},
            feature_hashes={self.ids[i]:digest(x) for i,x in enumerate(self.features)},
            imputed_values={self.ids[i]:int((~np.isfinite(x)).sum()) for i,x in enumerate(self.raw)},
            protocol='HD-SOH-v1; train-only scaling; no label/outlier cleaning; original row indices'))

    def checks(self):
        sets = [set(v) for v in self.splits.values()]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
        b = self.splits['test'][0]
        raw = self.raw[b].copy()
        original = normalize_features(raw, self.low, self.span, self.med)
        raw[21:] += 1000
        altered = normalize_features(raw, self.low, self.span, self.med)
        assert np.array_equal(original[:21], altered[:21])
        assert torch.equal(period_matrix(window_at(original, 20)), period_matrix(window_at(altered, 20)))
        old = self.labels[b].copy()
        self.labels[b] = old * 3
        assert np.array_equal(normalize_features(self.raw[b], self.low, self.span, self.med), original)
        self.labels[b] = old
        assert self.representation(b, 20, True).shape == (2,16,8,8)
        assert self.representation(b, 20, False).shape == (1,16,16,1)
        return dict(battery_disjoint=True, future_perturbation_invariant=True, test_label_input_invariant=True, shapes_verified=True)


class Arm:
    def __init__(self, name):
        seed_all()
        self.name = name
        if name.startswith('W'):
            self.u = Solution_u() if name == 'W-O' else hd.SolutionU(2048)
            # Same existing dynamics class and initialization policy in both arms.
            self.f = MLP(35, 1, 3, 60, .2)
            self.modules = [self.u, self.f]
            self.opts = [torch.optim.Adam(self.u.parameters(), lr=.002), torch.optim.Adam(self.f.parameters(), lr=.001)]
            self.scheduler = LR_Scheduler(self.opts[0],30,.002,200,.01,.0002)
        else:
            self.z = hd.ZhangSOH(1 if name == 'Z-O' else 2,16)
            self.modules = [self.z]
            self.opts = [torch.optim.Adam(self.z.parameters(), lr=.001)]
            self.scheduler = None

    def mode(self, training):
        for m in self.modules:
            m.train(training)

    def forward_wang(self, xt, rep):
        xt = xt.detach().requires_grad_(True)
        x,t = xt[:,:-1],xt[:,-1:]
        inp = torch.cat([x,t],1) if rep is None else torch.cat([rep.detach().flatten(1),x,t],1)
        u = self.u(inp)
        ut = torch.autograd.grad(u.sum(),t,create_graph=True)[0]
        ux = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        return u,ut-self.f(torch.cat([xt,u,ux,ut],1))

    def step(self, data, keys, rng):
        self.mode(True)
        for opt in self.opts:
            opt.zero_grad(set_to_none=True)
        x1,r1,y1 = data.batch(keys,self.name)
        x2,r2,y2 = data.batch(keys,self.name,True)
        mse = torch.nn.functional.mse_loss
        if self.name.startswith('W'):
            u1,f1 = self.forward_wang(x1,r1)
            u2,f2 = self.forward_wang(x2,r2)
            a = .5*mse(u1,y1)+.5*mse(u2,y2)
            b = .5*f1.square().mean()+.5*f2.square().mean()
            c = torch.relu((u2-u1)*(y1-y2)).sum()
            loss = a+.5*b+.01*c
            parts = dict(data=float(a.detach()),pde=float(b.detach()),physics=float(c.detach()))
        else:
            refs = data.references(keys,rng,1)[:,0]
            q1,yr1 = data.ref_batch(keys,refs,self.name)
            q2,yr2 = data.ref_batch(keys,refs,self.name,True)
            a = .5*mse(self.z.intra(r1),y1)+.5*mse(self.z.intra(r2),y2)
            b = .5*mse(self.z.delta(r1,q1),y1-yr1)+.5*mse(self.z.delta(r2,q2),y2-yr2)
            loss = a+b
            parts = dict(intra=float(a.detach()),delta=float(b.detach()))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss: '+self.name)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for m in self.modules for p in m.parameters())
        for opt in self.opts:
            opt.step()
        return dict(total=float(loss.detach()),**parts)

    @torch.no_grad()
    def predict(self,data,keys,rng):
        self.mode(False)
        xt,rep,y = data.batch(keys,self.name)
        if self.name.startswith('W'):
            pred = self.u(xt if rep is None else torch.cat([rep.flatten(1),xt],1))
        else:
            refs = data.references(keys,rng,2)
            intra = self.z.intra(rep)
            ps = []
            for j in range(2):
                ref,yr = data.ref_batch(keys,refs[:,j],self.name)
                ps.append(self.z.mixture(intra,self.z.delta(rep,ref),yr,.5))
            pred = torch.stack(ps).mean(0)
        assert pred.shape == y.shape and torch.isfinite(pred).all()
        return y.numpy(),pred.numpy()

    def states(self):
        return [m.state_dict() for m in self.modules]


def preflight(data, dest):
    save_json(dest/'checks.json',data.checks())
    results = []
    keys = data.pairs['train']
    order = np.random.default_rng(SEED).permutation(len(keys))
    for name in ARMS:
        model = Arm(name)
        rng = np.random.default_rng(10042)
        # Warm all benchmark batches once. Matrix preparation is separately timed.
        batches = [keys[order[i*BATCH:(i+1)*BATCH]] for i in range(9)]
        prepare_start = time.perf_counter()
        for batch in batches:
            data.batch(batch,name);data.batch(batch,name,True)
            if name.startswith('Z'):
                refs = data.references(batch,np.random.default_rng(10042),1)[:,0]
                data.ref_batch(batch,refs,name);data.ref_batch(batch,refs,name,True)
        prep = time.perf_counter()-prepare_start
        times=[]
        for i,batch in enumerate(batches):
            start=time.perf_counter()
            loss=model.step(data,batch,rng)
            elapsed=time.perf_counter()-start
            if i:times.append(elapsed)
        vt=[]
        vrng=np.random.default_rng(20042)
        for i in range(5):
            batch=data.pairs['valid'][i*BATCH:(i+1)*BATCH]
            start=time.perf_counter();model.predict(data,batch,vrng);vt.append(time.perf_counter()-start)
        train_s=float(np.quantile(times,.9))*math.ceil(len(keys)/BATCH)*EPOCHS
        valid_s=float(np.quantile(vt,.9))*math.ceil(len(data.pairs['valid'])/BATCH)*EPOCHS
        # Include a 25% margin for full-epoch materialization, host load and I/O.
        estimate=1.25*(train_s+valid_s)
        row=dict(arm=name,parameters=sum(p.numel() for m in model.modules for p in m.parameters()),
                 diagnostic_updates=9,warmup_batches=1,timed_batches=8,
                 train_batch_seconds=times,valid_batch_seconds=vt,preparation_seconds=prep,
                 projected_training_seconds=train_s,projected_validation_seconds=valid_s,
                 conservative_total_seconds=estimate,last_diagnostic_loss=loss)
        results.append(row)
        print(json.dumps({k:row[k] for k in ['arm','parameters','projected_training_seconds','projected_validation_seconds','conservative_total_seconds']}),flush=True)
        save_json(dest/'runtime_estimates.json',dict(arms=results,complete=False))
    total=sum(r['conservative_total_seconds'] for r in results)
    report=dict(arms=results,total_seconds=total,complete=True,within_one_hour=total<=3600,
                method='90th percentile timed batches times full split batches times 200 epochs; 25% margin; diagnostic models discarded',
                formal_training_started=False)
    save_json(dest/'runtime_estimates.json',report)
    return report


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    dest=OUT/'preflight'
    dest.mkdir(parents=True,exist_ok=False)
    print('Loading HD-SOH-v1 MIT, no formal training yet.',flush=True)
    data=Data();data.write_records(dest)
    save_json(dest/'config.json',dict(seed=SEED,epochs=EPOCHS,batch_size=BATCH,threads=8,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Diagnostic updates only. Four architectures and protocol fixed by existing analysis plan.'))
    print('Pair counts: '+str({k:len(v) for k,v in data.pairs.items()}),flush=True)
    report=preflight(data,dest)
    lines=['# MIT high-dimensional comparison: preflight','',
           'Only bounded diagnostic updates were executed; no formal results exist yet.','',
           '| Arm | Train estimate (min) | Validation estimate (min) | Total with margin (min) |',
           '| --- | ---: | ---: | ---: |']
    for r in report['arms']:
        lines.append(f"| {r['arm']} | {r['projected_training_seconds']/60:.1f} | {r['projected_validation_seconds']/60:.1f} | {r['conservative_total_seconds']/60:.1f} |")
    lines += ['',f"Total estimated: {report['total_seconds']/60:.1f} minutes.",
              'Estimates include validation every epoch and a 25% margin; not measurements of completed full runs.',
              'Existing 2-epoch smoke results are not used. Diagnostic weights will not initialize formal runs.',
              'Runtime artifacts, split/scaler/hash manifests and checks are in preflight/.',
              'Formal comparison_metrics.csv, per_battery_metrics.csv and prediction plots are intentionally not fabricated.']
    if not report['within_one_hour']:
        lines+=['','STOPPED: estimated full comparison exceeds the user\'s one-hour limit. Await revised runtime authorization; do not alter protocol or epochs to fit the budget.']
    (OUT/'ANALYSIS_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
