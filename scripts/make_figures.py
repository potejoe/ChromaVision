from pathlib import Path
import sys,json,tempfile,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chromavision.core import ColorModel,read_pair,perturb,stable_seed,CONDITIONS

OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.dpi':200,'font.family':'DejaVu Sans'})
COLORS=['#7a8793','#246b8e','#589a8d','#b68638','#955d8e','#b94e48']
ORDER=['RGB-rule','Hist-Lab','Hist-HS','Hist-CrCb','GrayWorld-Lab','Lab-Morph']
P=pd.read_csv(ROOT/'results/predictions.csv');S=pd.read_csv(ROOT/'results/summary.csv')
M=pd.read_csv(ROOT/'results/manifest_split.csv')

def save(fig,name):
    # Écriture atomique : un lecteur n'observe jamais un PNG partiellement écrit.
    with tempfile.NamedTemporaryFile(suffix='.png',dir=OUT,delete=False) as tmp:
        temp=Path(tmp.name)
    try:
        fig.savefig(temp,bbox_inches='tight');plt.close(fig)
        from PIL import Image
        with Image.open(temp) as image:image.verify()
        os.replace(temp,OUT/(name+'.png'))
    finally:
        temp.unlink(missing_ok=True)

fig,ax=plt.subplots(figsize=(7,3.6))
b=pd.read_csv(ROOT/'results/bootstrap_iou.csv');b=b[b.condition=='original'].set_index('method').loc[ORDER]
ax.barh(ORDER,b.mean_iou,color=COLORS,xerr=[b.mean_iou-b.low,b.high-b.mean_iou],capsize=3)
ax.set(xlim=(0,1),xlabel='IoU moyenne par image (15 images de test)',title='Condition originale et IC bootstrap à 95 %')
ax.invert_yaxis();save(fig,'clean_iou')

fig,ax=plt.subplots(figsize=(7.2,3.7))
grid=S.pivot(index='method',columns='condition',values='iou').loc[ORDER,CONDITIONS]
im=ax.imshow(grid,vmin=0,vmax=.6,cmap='viridis',aspect='auto');ax.set_xticks(range(6),['Orig.','Sombre','Claire','Chaud','Bruit','JPEG'])
ax.tick_params(axis='x',labelsize=9)
ax.set_yticks(range(6),ORDER)
for i in range(6):
    for j in range(6):ax.text(j,i,f'{grid.iloc[i,j]:.3f}',ha='center',va='center',color='white' if grid.iloc[i,j]<.3 else 'black')
fig.colorbar(im,ax=ax,label='IoU moyenne');save(fig,'robustness')

fig,ax=plt.subplots(figsize=(6.4,3.6))
for method,color in zip(ORDER,COLORS):
    s=S[(S.method==method)&(S.condition=='original')].iloc[0]
    ax.scatter(s.median_ms,s.iou,s=75,color=color,label=method)
ax.set(xlabel='Médiane de latence par image (ms, CPU, 1 thread)',ylabel='IoU moyenne',ylim=(.32,.48),title='Qualité et coût de prédiction, hors chargement')
ax.legend(fontsize=8,ncol=2);save(fig,'quality_latency')

fig,ax=plt.subplots(figsize=(6.5,3.6))
e=pd.read_csv(ROOT/'results/data_efficiency.csv')
for seed,g in e.groupby('seed'):
    a=g.groupby('budget').iou.mean();ax.plot(a.index,a.values,'o-',alpha=.55,label=f'Sous-échantillonnage {seed}')
ax.set(xlabel="Nombre d'images d'apprentissage",ylabel='IoU moyenne de test',xticks=[4,12,24,42],title='Histogramme Lab, réglages choisis sans utiliser le test')
ax.legend(fontsize=8);save(fig,'data_efficiency')

fig,ax=plt.subplots(figsize=(6.8,3.2))
v=pd.read_csv(ROOT/'results/validation_grid.csv')
for name,c in zip(['Hist-Lab','Hist-HS','Hist-CrCb','GrayWorld-Lab'],COLORS[1:5]):
    g=v[(v.method==name)&(v.bins==32)&(v.smooth==0)&(v.kernel==0)]
    ax.plot(g.threshold,g.val_iou,label=name,color=c)
ax.set(xlabel='Seuil sur le score',ylabel='IoU moyenne de validation',title='Choix du seuil sur 14 images de validation originales')
ax.legend(ncol=2,fontsize=8);save(fig,'validation_thresholds')

fig,ax=plt.subplots(figsize=(6.4,3))
t=M.groupby(['split','category']).size().unstack().reindex(['train','validation','test'])
t.plot.bar(stacked=True,ax=ax,color=['#246b8e','#b68638'],rot=0)
ax.set(xlabel='',ylabel="Nombre d'images",title='Séparation par photographie, stratifiée par type de scène')
ax.legend(title='',fontsize=8);save(fig,'split')

# Meilleur, médian et pire exemple du modèle déclaré Hist-Lab : pas seulement des succès.
g=P[(P.method=='Hist-Lab')&(P.condition=='original')].sort_values('iou')
selected=[g.iloc[0].id,g.iloc[len(g)//2].id,g.iloc[-1].id]
fig,axes=plt.subplots(3,4,figsize=(9,7.2))
lab=ColorModel.load(ROOT/'results/models','Hist-Lab');morph=ColorModel.load(ROOT/'results/models','Lab-Morph')
for i,ident in enumerate(selected):
    record=M.set_index('id').loc[ident].to_dict();record['id']=ident
    rgb,truth=read_pair(ROOT,record)
    for j,(image,title) in enumerate([(rgb,'Photo'),(truth,'Annotation'),(lab.predict(rgb),'Hist-Lab'),(morph.predict(rgb),'Lab-Morph')]):
        axes[i,j].imshow(image,cmap=None if j==0 else 'gray',vmin=0,vmax=1 if j else None)
        axes[i,j].axis('off')
        if i==0:axes[i,j].set_title(title)
    axes[i,0].text(0,-.07,['Pire cas','Cas médian','Meilleur cas'][i]+f" (IoU Lab {g.set_index('id').loc[ident,'iou']:.3f})",transform=axes[i,0].transAxes,fontsize=8)
fig.tight_layout();save(fig,'qualitative_cases')

ident=selected[0];record=M.set_index('id').loc[ident].to_dict();record['id']=ident
rgb,truth=read_pair(ROOT,record)
fig,axes=plt.subplots(1,4,figsize=(11,3.4))
for ax,image,title in zip(axes,[rgb,truth,lab.predict(rgb),morph.predict(rgb)],['Photographie','Annotation','Hist-Lab','Lab-Morph']):
    ax.imshow(image,cmap=None if image.ndim==3 else 'gray');ax.set_title(title);ax.axis('off')
fig.tight_layout();save(fig,'qualitative_worst')

# Même image, mêmes masques, six transformations. Exemple choisi par IoU médiane.
ident=selected[1];record=M.set_index('id').loc[ident].to_dict();record['id']=ident
rgb,truth=read_pair(ROOT,record)
fig,axes=plt.subplots(2,3,figsize=(8.5,5.7))
for ax,condition in zip(axes.ravel(),CONDITIONS):
    ax.imshow(perturb(rgb,condition,stable_seed(2026,ident,condition)));ax.set_title(condition);ax.axis('off')
fig.tight_layout();save(fig,'perturbations')

# Cas de masque pour le notebook, sans superposition ambiguë.
fig,ax=plt.subplots(figsize=(5,3));ax.imshow(truth,cmap='gray');ax.axis('off');ax.set_title('Annotation binaire fournie, blanc = région cutanée');save(fig,'mask_example')
print(OUT)
