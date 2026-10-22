"""Entraîner sur train, choisir sur validation propre, tester une fois le protocole figé."""
from pathlib import Path
import os
for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[var]='1'
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse, json, platform, time, hashlib, copy
import cv2, numpy as np, pandas as pd, scipy
from chromavision.core import (ColorModel, fit_histogram, binary_metrics, read_pair,
                               cleanup, CONDITIONS, perturb, stable_seed)
cv2.setNumThreads(1)

THRESHOLDS=np.linspace(.05,.95,19)


def split_records(records, seed=2026):
    result=[]
    for category in ['FacePhoto','FamilyPhoto']:
        group=sorted([r for r in records if r['category']==category],key=lambda r:r['id'])
        idx=np.random.default_rng(stable_seed(seed,category)).permutation(len(group))
        ntrain=int(.6*len(group));nval=int(.2*len(group))
        for pos,i in enumerate(idx):
            r=dict(group[i]);r['split']='train' if pos<ntrain else ('validation' if pos<ntrain+nval else 'test')
            result.append(r)
    return sorted(result,key=lambda r:r['id'])


def select_threshold(model, validation, kernels=(0,)):
    scores=[(mask,model.score(rgb)) for _,rgb,mask in validation]
    results=[]
    for k in kernels:
        for threshold in THRESHOLDS:
            ious=[binary_metrics(mask,cleanup(score>=threshold,k))['iou'] for mask,score in scores]
            results.append(dict(threshold=float(threshold),kernel=int(k),val_iou=float(np.mean(ious))))
    # Égalités résolues par le premier candidat (petit noyau, seuil croissant).
    best=max(results,key=lambda r:r['val_iou'])
    model.threshold=best['threshold'];model.kernel=best['kernel']
    return results,best


def tune(train,validation,out):
    models=[ColorModel('RGB-rule')];grid=[]
    specs=[('Hist-Lab','Lab',False),('Hist-HS','HS',False),('Hist-CrCb','CrCb',False),
           ('GrayWorld-Lab','Lab',True)]
    for name,space,gw in specs:
        candidates=[]
        for bins in [16,32]:
            for smooth in [0.,1.]:
                model=fit_histogram(train,name,space,bins,smooth,gw)
                rows,best=select_threshold(model,validation)
                grid += [dict(method=name,bins=bins,smooth=smooth,**r) for r in rows]
                candidates.append((best['val_iou'],model))
        selected=max(candidates,key=lambda x:x[0])[1];models.append(selected)
        print('Choix validation',name,selected.bins,selected.smooth,selected.threshold,flush=True)
    lab=next(m for m in models if m.name=='Hist-Lab')
    morph=copy.deepcopy(lab);morph.name='Lab-Morph'
    rows,best=select_threshold(morph,validation,kernels=(0,3,5))
    grid += [dict(method=morph.name,bins=morph.bins,smooth=morph.smooth,**r) for r in rows]
    models.append(morph)
    pd.DataFrame(grid).to_csv(out/'validation_grid.csv',index=False)
    for model in models:model.save(out/'models')
    return models


def evaluate(models,test_records,pairs,out,repeats=3):
    rows=[]
    for record in test_records:
        rgb,mask=pairs[record['id']]
        for condition in CONDITIONS:
            x=perturb(rgb,condition,stable_seed(2026,record['id'],condition))
            order=np.random.default_rng(stable_seed(record['id'],condition)).permutation(len(models))
            for j in order:
                model=models[j];model.predict(x) # échauffement hors chronométrage
                times=[]
                for _ in range(repeats):
                    start=time.perf_counter_ns();prediction=model.predict(x)
                    times.append((time.perf_counter_ns()-start)/1e6)
                rows.append(dict(id=record['id'],category=record['category'],condition=condition,
                    method=model.name,height=x.shape[0],width=x.shape[1],model_bytes=model.nbytes,
                    median_ms=float(np.median(times)),**binary_metrics(mask,prediction)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'predictions.csv',index=False)
    timing_summary=frame.groupby(['method','condition'],sort=False).agg(
        iou=('iou','mean'),dice=('dice','mean'),precision=('precision','mean'),
        recall=('recall','mean'),fpr=('fpr','mean'),median_ms=('median_ms','median'),n=('id','size')).reset_index()
    timing_summary.to_csv(out/'summary.csv',index=False)
    # Bootstrap par image, mêmes indices entre méthodes et conditions.
    ids=sorted(frame['id'].unique());rng=np.random.default_rng(2026)
    resamples=rng.integers(0,len(ids),size=(3000,len(ids)))
    boot=[]
    for (method,condition),g in frame.groupby(['method','condition']):
        values=g.set_index('id').loc[ids,'iou'].to_numpy()
        lo,hi=np.quantile(values[resamples].mean(1),[.025,.975])
        boot.append(dict(method=method,condition=condition,mean_iou=float(values.mean()),low=float(lo),high=float(hi)))
    pd.DataFrame(boot).to_csv(out/'bootstrap_iou.csv',index=False)
    # Effets appariés prévus : morphologie contre Lab, espaces couleur contre règle RGB.
    contrasts=[]
    for a,b in [('Hist-Lab','RGB-rule'),('Hist-HS','RGB-rule'),('Hist-CrCb','RGB-rule'),
                ('GrayWorld-Lab','Hist-Lab'),('Lab-Morph','Hist-Lab')]:
        for condition in CONDITIONS:
            get=lambda method: frame[(frame.method==method)&(frame.condition==condition)].set_index('id').loc[ids,'iou'].to_numpy()
            values=get(a)-get(b);lo,hi=np.quantile(values[resamples].mean(1),[.025,.975])
            contrasts.append(dict(a=a,b=b,condition=condition,delta=float(values.mean()),low=float(lo),high=float(hi)))
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    robustness=[]
    def get_values(method,condition):
        return frame[(frame.method==method)&(frame.condition==condition)].set_index('id').loc[ids,'iou'].to_numpy()
    for method in [m.name for m in models if m.name!='RGB-rule']:
        for condition in CONDITIONS[1:]:
            values=((get_values(method,condition)-get_values(method,'original'))
                    -(get_values('RGB-rule',condition)-get_values('RGB-rule','original')))
            lo,hi=np.quantile(values[resamples].mean(1),[.025,.975])
            robustness.append(dict(method=method,condition=condition,
                delta_relative_to_rgb=float(values.mean()),low=float(lo),high=float(hi)))
    pd.DataFrame(robustness).to_csv(out/'robustness_contrasts.csv',index=False)
    return frame


def efficiency(train,validation,test,lab,out):
    rows=[]
    for budget in [4,12,24,len(train)]:
        for seed in [17,28,39]:
            order=np.random.default_rng(seed).permutation(len(train))[:budget]
            subset=[train[i] for i in order]
            start=time.perf_counter()
            # Refaire tout le choix de réglages pour chaque budget : aucun modèle
            # appris sur les 42 images n'est utilisé pour le budget réduit.
            candidates=[]
            for bins in [16,32]:
                for smooth in [0.,1.]:
                    candidate=fit_histogram(subset,'Hist-Lab','Lab',bins,smooth)
                    _,selection=select_threshold(candidate,validation)
                    candidates.append((selection['val_iou'],candidate))
            model=max(candidates,key=lambda c:c[0])[1]
            fit_s=time.perf_counter()-start
            for ident,rgb,mask in test:
                rows.append(dict(budget=budget,seed=seed,id=ident,fit_seconds=fit_s,
                    threshold=model.threshold,bins=model.bins,smooth=model.smooth,
                    iou=binary_metrics(mask,model.predict(rgb))['iou']))
    pd.DataFrame(rows).to_csv(out/'data_efficiency.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--max-side',type=int,default=320);a=p.parse_args()
    root=a.root.resolve();out=root/'results';out.mkdir(parents=True,exist_ok=True)
    raw=json.loads((root/'data/manifest.json').read_text())
    records=split_records(raw['records'])
    pd.DataFrame(records).to_csv(out/'manifest_split.csv',index=False)
    # Enregistrer le plan avant de lire les étiquettes de test pour l'évaluation.
    plan=dict(seed=2026,max_side=a.max_side,bins=[16,32],smooth=[0,1],thresholds=THRESHOLDS.tolist(),
              conditions=CONDITIONS,timing_repeats=3,bootstrap=3000,primary='macro IoU par image',
              criterion='IoU moyenne sur validation originale uniquement',deep_learning=False,
              variants=['RGB-rule','Hist-Lab','Hist-HS','Hist-CrCb','GrayWorld-Lab','Lab-Morph'],
              data_budgets=[4,12,24,42],data_seeds=[17,28,39],
              data_budget_tuning='réestimer bins, lissage et seuil par budget sur les mêmes 14 images de validation')
    (out/'protocol.json').write_text(json.dumps(plan,indent=2))
    pairs={r['id']:read_pair(root,r,a.max_side) for r in records}
    get=lambda split:[(r['id'],*pairs[r['id']]) for r in records if r['split']==split]
    train,validation,test=get('train'),get('validation'),get('test')
    print('Séparation',len(train),len(validation),len(test),flush=True)
    start=time.perf_counter();models=tune(train,validation,out);fit_seconds=time.perf_counter()-start
    # Empreinte des modèles/configurations figées AVANT l'évaluation test.
    locked={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((out/'models').iterdir())}
    (out/'model_lock.json').write_text(json.dumps(locked,indent=2))
    evaluate(models,[r for r in records if r['split']=='test'],pairs,out)
    efficiency(train,validation,test,next(m for m in models if m.name=='Hist-Lab'),out)
    env=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
             numpy=np.__version__,opencv=cv2.__version__,pandas=pd.__version__,scipy=scipy.__version__,
             cv_threads=cv2.getNumThreads(),fit_and_validation_seconds=fit_seconds,
             seed=2026,command='python scripts/run_experiments.py',test_reuse='tous les contrastes planifiés ; pas de sélection sur test')
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    print('Expériences terminées',out,flush=True)


if __name__=='__main__':main()
