"""Vérifier séparations, empreintes des modèles et recalcul des métriques exportées."""
from pathlib import Path
import json,hashlib,sys,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--with-data',action='store_true',help='Vérifier aussi les empreintes des photographies et des masques locaux')
args=parser.parse_args()
r=pd.read_csv(ROOT/'results/manifest_split.csv')
assert r.id.is_unique and r.sha256_pixels.is_unique
assert r.groupby('split').size().to_dict()=={'test':15,'train':42,'validation':14}
if args.with_data:
    for row in r.to_dict('records'):
        for field in ['image','mask']:
            assert hashlib.sha256((ROOT/row[field]).read_bytes()).hexdigest()==row['sha256_'+field]
p=pd.read_csv(ROOT/'results/predictions.csv')
assert len(p)==15*6*6 and not p.duplicated(['id','condition','method']).any()
assert set(p.id)==set(r.loc[r.split=='test','id'])
np.testing.assert_allclose(p.iou,p.tp/(p.tp+p.fp+p.fn),atol=1e-12)
np.testing.assert_allclose(p.dice,2*p.tp/(2*p.tp+p.fp+p.fn),atol=1e-12)
s=pd.read_csv(ROOT/'results/summary.csv')
calc=p.groupby(['method','condition']).iou.mean()
np.testing.assert_allclose(s.set_index(['method','condition']).iou.sort_index(),calc.sort_index())
locked=json.loads((ROOT/'results/model_lock.json').read_text())
for name,digest in locked.items():assert hashlib.sha256((ROOT/'results/models'/name).read_bytes()).hexdigest()==digest
v=pd.read_csv(ROOT/'results/validation_grid.csv');assert len(v)==361
e=pd.read_csv(ROOT/'results/data_efficiency.csv');assert len(e)==4*3*15
assert sorted(e.budget.unique())==[4,12,24,42]
rob=pd.read_csv(ROOT/'results/robustness_contrasts.csv')
assert len(rob)==25
ids=sorted(p.id.unique())
def get(m,c):return p[(p.method==m)&(p.condition==c)].set_index('id').loc[ids].iou.to_numpy()
for row in rob.to_dict('records'):
    calc=(get(row['method'],row['condition'])-get(row['method'],'original'))-(get('RGB-rule',row['condition'])-get('RGB-rule','original'))
    assert abs(calc.mean()-row['delta_relative_to_rgb'])<1e-12
print('Manifeste : 71 paires, séparation 42/14/15. 540 prédictions et métriques cohérentes. Empreintes des modèles vérifiées.')
print('Empreintes des photographies et des masques vérifiées.' if args.with_data else 'Photographies absentes du dépôt public : empreintes des fichiers source non vérifiées. Utiliser --with-data après import local.')
