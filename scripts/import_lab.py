"""Importer uniquement Pratheepan depuis le ZIP du TP, sans extraction aveugle."""
from pathlib import Path
import argparse, hashlib, io, json, zipfile
import cv2
import numpy as np


def import_lab(source, dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as outer:
        nested = [n for n in outer.namelist() if Path(n).name == 'lab1_student.zip']
        blob = io.BytesIO(outer.read(nested[0])) if nested else None
        archive = zipfile.ZipFile(blob) if blob else outer
        images, masks = {}, {}
        for name in archive.namelist():
            path = Path(name)
            if 'Face_Dataset' not in name or path.suffix.lower() not in ('.jpg','.jpeg','.png'):
                continue
            # Noms de fichiers uniquement : impossible de sortir du dossier cible.
            if 'Pratheepan_Dataset/' in name:
                images[path.stem.lower()] = (name, archive.read(name))
            elif 'Ground_Truth/' in name:
                masks[path.stem.lower()] = (name, archive.read(name))
        records=[]
        for key,(mask_name, mask_bytes) in sorted(masks.items()):
            # Correction de nom documentée par le site officiel en mars 2017.
            image_key = '1_bo_allfamily_lrg' if key == 'obama' and key not in images else key
            if image_key not in images:
                raise ValueError(f'Photo manquante pour {mask_name}')
            image_name, image_bytes = images[image_key]
            category = 'FamilyPhoto' if '/FamilyPhoto/' in image_name else 'FacePhoto'
            ident = category+'__'+key
            ip=Path('data/raw/images')/(ident+Path(image_name).suffix.lower())
            mp=Path('data/raw/masks')/(ident+'.png')
            for rel,data in [(ip,image_bytes),(mp,mask_bytes)]:
                target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            im=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR)
            ma=cv2.imdecode(np.frombuffer(mask_bytes,np.uint8),cv2.IMREAD_GRAYSCALE)
            if im is None or ma is None or im.shape[:2] != ma.shape:
                raise ValueError(f'Paire invalide : {ident}')
            records.append(dict(id=ident,category=category,image=str(ip),mask=str(mp),
                height=im.shape[0],width=im.shape[1],positive_fraction=float((ma>=128).mean()),
                sha256_image=hashlib.sha256(image_bytes).hexdigest(),
                sha256_mask=hashlib.sha256(mask_bytes).hexdigest(),
                sha256_pixels=hashlib.sha256(im.tobytes()).hexdigest()))
    if len({r['sha256_pixels'] for r in records}) != len(records):
        raise ValueError('Doublon exact de pixels : créer des groupes avant de séparer')
    (dest/'data').mkdir(exist_ok=True)
    (dest/'data/manifest.json').write_text(json.dumps(dict(
        origin='Sous-ensemble Pratheepan fourni dans le TP Lab 1',
        source_archive_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        records=records),indent=2))
    print(f'{len(records)} paires importées dans {dest}')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--dest',default='.')
    a=p.parse_args();import_lab(a.source,a.dest)
