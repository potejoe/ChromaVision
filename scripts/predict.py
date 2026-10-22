"""Inférence sur une photo personnelle avec un modèle déjà entraîné."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2
from chromavision.core import ColorModel

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--method',default='Hist-Lab')
    p.add_argument('--output',default='prediction.png');a=p.parse_args()
    bgr=cv2.imread(a.image)
    if bgr is None:raise SystemExit('Image illisible : '+a.image)
    scale=min(1.,320/max(bgr.shape[:2]))
    if scale<1:bgr=cv2.resize(bgr,(round(bgr.shape[1]*scale),round(bgr.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    pred=ColorModel.load(ROOT/'results/models',a.method).predict(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    if not cv2.imwrite(a.output,pred.astype('uint8')*255):raise SystemExit('Écriture impossible')
    print(a.output)
