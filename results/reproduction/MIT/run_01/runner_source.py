"""One author-protocol MIT run; instrumentation only, no replacement training loop."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from main_MIT import get_args, load_MIT_data
from Model.Model import PINN
from scripts.smoke_test import mit_split_manifest

OUT = ROOT / 'results/reproduction/MIT/run_01'

def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')

def metrics(y, p):
    mse = float(mean_squared_error(y, p))
    return dict(MAE=float(mean_absolute_error(y, p)),
                MAPE_fraction=float(mean_absolute_percentage_error(y, p)),
                MAPE_percent=float(100*mean_absolute_percentage_error(y, p)),
                RMSE=float(np.sqrt(mse)), MSE=mse)

class RecordedPINN(PINN):
    """Calls original training, validation and testing exactly once at original times."""
    def __init__(self, args):
        super().__init__(args)
        self.history = []
        self.test_history = []
        self.best_epoch_recorded = None

    def train_one_epoch(self, epoch, dataloader):
        start = time.perf_counter()
        lr = self.optimizer1.param_groups[0]['lr']
        losses = super().train_one_epoch(epoch, dataloader)
        if not np.isfinite(losses).all():
            raise RuntimeError('Nonfinite training losses')
        a, b, c = losses
        self.history.append(dict(epoch=epoch, learning_rate_used=lr,
            data_loss=a, PDE_loss=b, physics_loss=c,
            total_loss=a+self.alpha*b+self.beta*c,
            training_seconds=time.perf_counter()-start))
        return losses

    def Valid(self, loader):
        value = super().Valid(loader)
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite validation MSE')
        self.history[-1]['validation_MSE'] = value
        self.history[-1]['learning_rate_next'] = self.optimizer1.param_groups[0]['lr']
        with (OUT / 'loss_history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[-1]))
            writer.writeheader()
            writer.writerows(self.history)
        return value

    def Test(self, loader):
        y, p = super().Test(loader)
        if not np.isfinite(p).all():
            raise RuntimeError('Nonfinite test predictions')
        epoch = self.history[-1]['epoch']
        self.best_epoch_recorded = epoch
        # Save immediately; unlike the author's deferred shallow state_dict this
        # file preserves the parameters at the actual best validation epoch.
        torch.save(dict(solution_u=self.solution_u.state_dict(),
                        dynamical_F=self.dynamical_F.state_dict(), epoch=epoch,
                        validation_MSE=self.history[-1]['validation_MSE']),
                   OUT / 'best_checkpoint.pth')
        self.test_history.append(dict(epoch=epoch, **metrics(y, p)))
        write_json('test_history.json', self.test_history)
        return y, p

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    class Tee:
        def __init__(self, original, log):
            self.original, self.log = original, log
        def write(self, text):
            self.original.write(text)
            self.log.write(text)
            self.log.flush()
        def flush(self):
            self.original.flush()
            self.log.flush()
    console = (OUT/'console.log').open('w', encoding='utf-8')
    sys.stdout, sys.stderr = Tee(sys.stdout, console), Tee(sys.stderr, console)
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    # No manual seed: retain the original author's unseeded initialization.
    args = get_args()
    args.save_folder = str(OUT)
    assert (args.epochs, args.batch_size, args.alpha, args.beta) == (200, 512, .5, .01)
    source_paths = ['main_MIT.py', 'Model/Model.py', 'dataloader/dataloader.py',
                    'utils/util.py', 'scripts/reproduce_mit_one.py']
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    git = lambda *a: subprocess.check_output(['git', *a], text=True, encoding='utf-8')
    (OUT/'source_diff.patch').write_text(git('diff', 'HEAD'), encoding='utf-8')
    (OUT/'source_status.txt').write_text(git('status', '--short'), encoding='utf-8')
    shutil.copy2(__file__, OUT/'runner_source.py')
    shutil.copy2(ROOT/'requirements-reproduction.txt', OUT/'requirements-reproduction.txt')
    torch.save(dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state()), OUT/'rng_initial.pth')
    config = dict(arguments=vars(args), experiment_count=1, started_utc=started,
        source_commit=git('rev-parse', 'HEAD').strip(), source_sha256=hashes,
        python=platform.python_version(), torch=torch.__version__,
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        seed_policy='Original global RNG unseeded; split random_state=420; initial RNG states saved',
        split=mit_split_manifest(),
        instrumentation='Original PINN.Train unchanged. Subclass records original method returns and saves immediate best checkpoint. Standard metrics added separately; original log/model.pth retained.')
    write_json('run_configuration.json', config)
    loaders = load_MIT_data(args)
    sizes = {k: len(v.dataset) for k,v in loaders.items()}
    assert sizes == dict(train=54053, valid=13514, test=14170), sizes
    for loader in loaders.values():
        assert all(torch.isfinite(t).all() for t in loader.dataset.tensors)
    config['dataset_sizes'] = sizes
    config['tensor_sha256'] = {k: [hashlib.sha256(t.numpy().tobytes()).hexdigest()
                                      for t in v.dataset.tensors] for k,v in loaders.items()}
    model = RecordedPINN(args)
    config['device'] = str(next(model.parameters()).device)
    write_json('run_configuration.json', config)
    train_start = time.perf_counter()
    model.Train(trainloader=loaders['train'], validloader=loaders['valid'], testloader=loaders['test'])
    training_seconds = time.perf_counter()-train_start
    y = np.load(OUT/'true_label.npy')
    p = np.load(OUT/'pred_label.npy')
    best = torch.load(OUT/'best_checkpoint.pth', map_location='cpu', weights_only=True)
    model.solution_u.load_state_dict(best['solution_u'])
    model.dynamical_F.load_state_dict(best['dynamical_F'])
    # Call the original base method to verify saved weights, without triggering recording.
    check_y, check_p = PINN.Test(model, loaders['test'])
    assert np.array_equal(y, check_y)
    assert np.allclose(p, check_p, rtol=1e-6, atol=1e-7)
    stop = model.history[-1]['epoch']
    result = dict(status='complete', **metrics(y,p), test_samples=len(y),
        aggregation='sample-weighted; SOH fraction; standard ground-truth denominator MAPE',
        zero_targets=int(np.count_nonzero(y == 0)),
        author_MAPE_fraction=float(mean_absolute_percentage_error(p,y)),
        best_epoch=model.best_epoch_recorded, stopping_epoch=stop,
        best_validation_MSE=best['validation_MSE'],
        stop_reason='maximum epochs' if stop == args.epochs else 'original early_stop > 20',
        checkpoint_verified=True, maximum_prediction_difference=float(np.max(np.abs(p-check_p))),
        training_loop_seconds=training_seconds)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y.ravel()*100, p.ravel()*100, s=3, alpha=.2, rasterized=True)
    lo, hi = min(y.min(),p.min())*100, max(y.max(),p.max())*100
    ax.plot([lo,hi],[lo,hi], 'k--', linewidth=1)
    ax.set(xlabel='True SOH (%)', ylabel='Predicted SOH (%)',
           title=f'MIT run 01: best validation epoch {result["best_epoch"]}')
    fig.tight_layout(); fig.savefig(OUT/'predicted_vs_true_SOH.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for key in ['data_loss','PDE_loss','physics_loss','total_loss']:
        axes[0].semilogy([r['epoch'] for r in model.history], [r[key] for r in model.history], label=key)
    axes[1].semilogy([r['epoch'] for r in model.history], [r['validation_MSE'] for r in model.history])
    axes[0].legend(); axes[0].set(xlabel='Epoch',ylabel='Training loss')
    axes[1].set(xlabel='Epoch',ylabel='Validation MSE')
    fig.tight_layout(); fig.savefig(OUT/'loss_history.png', dpi=180); plt.close(fig)
    result['total_runtime_seconds'] = time.perf_counter()-start
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json('metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
