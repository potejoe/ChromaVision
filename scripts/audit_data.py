"""Contrôles d'intégrité et détecteur indicatif de quasi-doublons par dHash."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'data/manifest.json').read_text())['records']
hashes=[]
for r in records:
    assert hashlib.sha256((ROOT/r['image']).read_bytes()).hexdigest()==r['sha256_image']
    rgb=cv2.imread(str(ROOT/r['image']));mask=cv2.imread(str(ROOT/r['mask']),0)
    assert rgb.shape[:2]==mask.shape and np.any(mask>=128) and np.any(mask<128)
    tiny=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),(9,8))
    hashes.append((tiny[:,1:]>tiny[:,:-1]).ravel())
near=[]
for i in range(len(records)):
    for j in range(i):
        distance=int(np.count_nonzero(hashes[i]!=hashes[j]))
        if distance<=4:near.append(dict(a=records[i]['id'],b=records[j]['id'],distance=distance))
pd.DataFrame(near,columns=['a','b','distance']).to_csv(ROOT/'results/near_duplicates.csv',index=False)
audit=dict(pairs=len(records),exact_pixel_duplicates=len(records)-len({r['sha256_pixels'] for r in records}),
           dhash_pairs_distance_le_4=len(near),mask_threshold=128,
           note='dHash ne prouve pas l’absence de scènes ou de personnes communes ; pas de métadonnées d’identité')
(ROOT/'results/audit.json').write_text(json.dumps(audit,indent=2))
print(audit)
