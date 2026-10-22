"""RSS échantillonnée en processus isolés. Ce n'est ni une mesure d'énergie ni un pic exact."""
from pathlib import Path
import sys, os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse, json, subprocess, time, threading
import psutil
import cv2, numpy as np, pandas as pd
from chromavision.core import ColorModel,read_pair


def worker(method):
    cv2.setNumThreads(1)
    records=pd.read_csv(ROOT/'results/manifest_split.csv').to_dict('records')
    record=max((r for r in records if r['split']=='test'),key=lambda r:r['height']*r['width'])
    image,_=read_pair(ROOT,record)
    model=ColorModel.load(ROOT/'results/models',method)
    def rss():
        # /proc/self fonctionne aussi dans un espace de noms PID remappé.
        if Path('/proc/self/status').exists():
            line=next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:'))
            return int(line.split()[1])*1024
        return psutil.Process().memory_info().rss
    baseline=rss()
    samples=[baseline];stop=threading.Event()
    def sample():
        while not stop.is_set():
            samples.append(rss());stop.wait(.001)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    for _ in range(50):model.predict(image)
    stop.set();monitor.join()
    return dict(method=method,model_bytes=model.nbytes,input_bytes=image.nbytes,
                baseline_rss_mib=baseline/2**20,sampled_peak_rss_mib=max(samples)/2**20,
                sampled_delta_mib=(max(samples)-baseline)/2**20,repetitions=50,
                note='processus Python + OpenCV ; échantillonnage 1 ms ; peut manquer un pic bref')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args()
    if a.worker: print(json.dumps(worker(a.worker)))
    else:
        rows=[]
        for file in sorted((ROOT/'results/models').glob('*.json')):
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            r=subprocess.run([sys.executable,__file__,'--worker',file.stem],capture_output=True,text=True,check=True,env=env)
            rows.append(json.loads(r.stdout))
        pd.DataFrame(rows).to_csv(ROOT/'results/memory.csv',index=False)
        print(pd.DataFrame(rows).to_string(index=False))
