"""Read completed historical run and author archive; no model imports or training."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/reproduction/MIT/historical_run_01'

def metrics(y, p):
    mse = float(mean_squared_error(y, p))
    return dict(MAE=float(mean_absolute_error(y,p)),
                MAPE_percent=float(100*mean_absolute_percentage_error(y,p)),
                RMSE=float(np.sqrt(mse)), MSE=mse)

def historical_aggregation(y, p):
    # Mirror the archived analysis convention, including its boundary omissions
    # and reversed MAPE denominator. Do not present these as standard metrics.
    y, p = y.ravel(), p.ravel()
    ends = np.r_[np.where(np.diff(y) > .05)[0], len(y)]
    rows, start = [], 0
    for end in ends:
        yy, pp = y[start:end], p[start:end]
        row = metrics(yy, pp)
        row['MAPE_percent'] = float(100*mean_absolute_percentage_error(pp, yy))
        rows.append(row)
        start = end+1
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}

def main():
    result = json.loads((OUT/'metrics.json').read_text())
    assert result['status'] == 'complete'
    config = json.loads((OUT/'run_configuration.json').read_text())
    y, p = np.load(OUT/'true_label.npy'), np.load(OUT/'pred_label.npy')
    rows, historic_rows = [], []
    archive_configs = []
    with zipfile.ZipFile(ROOT/'results/Ours/MIT results.zip') as z:
        for i in range(1,11):
            base = f'MIT results/Experiment{i}/'
            ay = np.load(io.BytesIO(z.read(base+'true_label.npy')))
            ap = np.load(io.BytesIO(z.read(base+'pred_label.npy')))
            assert np.array_equal(ay, y), f'Archive {i} labels differ'
            text = z.read(base+'logging.txt').decode('utf-8')
            args = dict(line.split('\t')[-1].split(':',1) for line in text.splitlines() if 'CRITICAL' in line)
            archive_configs.append(args)
            for key, value in args.items():
                if key in ('save_folder','log_dir','data','normalization_method'):
                    continue
                assert float(value) == float(config['arguments'][key]), (i,key,value)
            paths = [ast.literal_eval(line) for line in text.splitlines() if line.startswith('[') and '.csv' in line]
            for archived, key in zip(paths, ['training_and_validation_battery_files','test_battery_files']):
                assert [v.replace('\\','/') for v in archived] == config['split'][key]
            rows.append(dict(experiment=i,**metrics(ay,ap)))
            historic_rows.append(dict(experiment=i,**historical_aggregation(ay,ap)))
    current = metrics(y,p)
    stats = {k:dict(mean=float(np.mean([r[k] for r in rows])),
                    std=float(np.std([r[k] for r in rows],ddof=1)),
                    min=min(r[k] for r in rows), max=max(r[k] for r in rows),
                    historical_run_01=current[k],
                    relative_difference_percent=float((current[k]/np.mean([r[k] for r in rows])-1)*100),
                    within_archive_range=bool(min(r[k] for r in rows)<=current[k]<=max(r[k] for r in rows)))
             for k in current}
    history = list(csv.DictReader((OUT/'loss_history.csv').open()))
    assert len(history) == result['stopping_epoch']
    best = min(history,key=lambda r:float(r['validation_MSE']))
    assert int(best['epoch']) == result['best_epoch']
    assert result['stopping_epoch'] == 200 or result['stopping_epoch']-result['best_epoch'] == 21
    assert all(np.isfinite(float(v)) for r in history for v in r.values())
    for r in history:
        assert abs(float(r['total_loss'])-(float(r['data_loss'])+float(r['PDE_loss'])+50*float(r['physics_loss']))) < 1e-10
    protected=json.loads((ROOT/'historical_reproduction/protected_before.json').read_text())
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in protected.items())
    report=dict(source_archive='results/Ours/MIT results.zip', archive_configurations=archive_configs,
        archive_runs_standard=rows, standard_metric_comparison=stats,
        archive_analysis_convention=dict(description='Equal battery weighting; heuristic boundary omissions; prediction-denominator MAPE.',
            run_01=historical_aggregation(y,p),
            archive_mean={k:float(np.mean([r[k] for r in historic_rows])) for k in current}),
        checks=dict(all_archive_numeric_arguments_match=True, all_archive_labels_bitwise_equal=True,
            all_archive_ordered_file_lists_equal=True, history_and_stopping_verified=True,
            current_code_and_run_01_unchanged=True),
        caveat='One unseeded run cannot establish the repeated-run distribution or bitwise historical reproducibility.')
    (OUT/'archive_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(stats,indent=2))
    print(json.dumps(report['archive_analysis_convention'],indent=2))

if __name__ == '__main__':
    main()
