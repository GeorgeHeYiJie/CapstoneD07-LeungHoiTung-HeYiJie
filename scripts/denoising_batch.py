"""Durable, sequential four-condition batch. Stops immediately on any error."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/denoising_analysis'

def write_status(**fields):
    fields.update(updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),supervisor_pid=os.getpid())
    temp=OUT/'batch_status.tmp';temp.write_text(json.dumps(fields,indent=2)+'\n',encoding='utf8');temp.replace(OUT/'batch_status.json')

def main():
    for test in ['test1','test2']:
        for condition in ['original','denoised']:
            dest=OUT/test/condition
            if (dest/'metrics.json').exists() and json.loads((dest/'progress.json').read_text()).get('complete'):
                continue
            args=[sys.executable,'-u',str(ROOT/'scripts/run_denoising_analysis.py'),'run','--test',test,'--condition',condition]
            if dest.exists():args.append('--resume')
            with (OUT/f'{test}_{condition}.log').open('a',encoding='utf8') as log:
                child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write_status(state='running',test=test,condition=condition,child_pid=child.pid,command=args)
                result=child.wait()
                if result:raise RuntimeError(f'{test}/{condition} failed, exit {result}; see log. No remaining conditions launched.')
        write_status(state='analyzing',test=test)
        with (OUT/f'{test}_analysis.log').open('a',encoding='utf8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/analyze_denoising_results.py'),'--test',test],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_status(state='complete',message='All four conditions and both analysis reports completed. No other experiments launched.')

if __name__=='__main__':
    try:main()
    except Exception:
        write_status(state='failed',error=traceback.format_exc())
        raise
