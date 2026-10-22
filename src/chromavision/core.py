"""Algorithmes sans accès aux annotations de test. Images RGB uint8, masques bool."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, values)).encode()).digest()[:4], 'little')


def read_pair(root, record, max_side=320):
    rgb = cv2.cvtColor(cv2.imread(str(Path(root)/record['image'])), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(Path(root)/record['mask']), cv2.IMREAD_GRAYSCALE)
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"Image/masque désalignés : {record['id']}")
    scale = min(1., max_side/max(rgb.shape[:2]))
    if scale < 1:
        size = (max(1, round(rgb.shape[1]*scale)), max(1, round(rgb.shape[0]*scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return rgb, mask >= 128


def gray_world(rgb):
    """Gains de canaux bornés. Le gris moyen est une hypothèse, pas une vérité."""
    x = rgb.astype(np.float32)
    means = x.mean(axis=(0, 1))
    gains = np.clip(means.mean()/np.maximum(means, 1.), .5, 2.)
    return np.clip(x*gains, 0, 255).astype(np.uint8)


def color_bins(rgb, space='Lab', bins=32):
    """Conversions uint8 OpenCV. H est dans [0,179], les autres axes dans [0,255]."""
    if space == 'Lab':
        x = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1:3]
        divisors = np.array([256., 256.])
    elif space == 'HS':
        x = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., :2]
        divisors = np.array([180., 256.])
    elif space == 'CrCb':
        x = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 1:3]
        divisors = np.array([256., 256.])
    else:
        raise ValueError('Espace attendu : Lab, HS ou CrCb')
    return np.minimum((x.astype(np.float32)*bins/divisors).astype(np.int32), bins-1)


def rgb_rule(rgb):
    """Règle RGB du TP, d'après Vezhnevets et al. (2003), domaine uint8."""
    x = rgb.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return ((r > 95) & (g > 40) & (b > 20) & ((x.max(-1)-x.min(-1)) > 15)
            & (np.abs(r-g) > 15) & (r > g) & (r > b))


def cleanup(mask, kernel=0):
    if not kernel:
        return mask.astype(bool)
    if kernel % 2 != 1 or kernel < 1:
        raise ValueError('Le noyau morphologique doit être impair et positif')
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, elem)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, elem).astype(bool)


@dataclass
class ColorModel:
    name: str
    space: str = 'Lab'
    bins: int = 32
    smooth: float = 1.
    threshold: float = .5
    kernel: int = 0
    white_balance: bool = False
    lut: object = None

    def score(self, rgb):
        if self.name == 'RGB-rule':
            return rgb_rule(rgb).astype(np.float32)
        if self.lut is None:
            raise RuntimeError('Modèle non appris')
        x = gray_world(rgb) if self.white_balance else rgb
        q = color_bins(x, self.space, self.bins)
        return self.lut[q[..., 0], q[..., 1]]

    def predict(self, rgb):
        return cleanup(self.score(rgb) >= self.threshold, self.kernel)

    @property
    def nbytes(self):
        return 0 if self.lut is None else self.lut.nbytes

    def save(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        meta = {k:v for k,v in self.__dict__.items() if k != 'lut'}
        (folder/(self.name+'.json')).write_text(json.dumps(meta, indent=2))
        if self.lut is not None:
            np.save(folder/(self.name+'.npy'), self.lut, allow_pickle=False)

    @classmethod
    def load(cls, folder, name):
        folder = Path(folder)
        model = cls(**json.loads((folder/(name+'.json')).read_text()))
        if name != 'RGB-rule':
            model.lut = np.load(folder/(name+'.npy'), allow_pickle=False)
        return model


def fit_histogram(pairs, name, space='Lab', bins=32, smooth=1., white_balance=False,
                  pixels_per_class=2500, seed=2026):
    """Même quota maximal par image/classe, lissage, a priori équilibré explicite.

    Les compteurs d'histogramme sont normalisés séparément. La LUT est donc un
    score bayésien sous a priori 0,5, pas une probabilité calibrée à la prévalence.
    """
    hist = np.full((2, bins, bins), .5, dtype=np.float64)
    for ident, rgb, mask in pairs:
        q = color_bins(gray_world(rgb) if white_balance else rgb, space, bins)
        flat = q.reshape(-1, 2)
        for label in (0, 1):
            idx = np.flatnonzero(mask.reshape(-1) == label)
            rng = np.random.default_rng(stable_seed(seed, ident, label))
            if len(idx) > pixels_per_class:
                idx = rng.choice(idx, pixels_per_class, replace=False)
            np.add.at(hist[label], (flat[idx, 0], flat[idx, 1]), 1.)
    for label in (0, 1):
        if smooth:
            mode = ('wrap', 'nearest') if space == 'HS' else 'reflect'
            hist[label] = gaussian_filter(hist[label], smooth, mode=mode)
        hist[label] /= hist[label].sum()
    lut = (hist[1]/(hist[0]+hist[1])).astype(np.float32)
    return ColorModel(name, space, bins, smooth, white_balance=white_balance, lut=lut)


def binary_metrics(truth, prediction):
    t, p = np.asarray(truth, bool), np.asarray(prediction, bool)
    if t.shape != p.shape:
        raise ValueError('Les masques doivent avoir la même taille')
    tp = int(np.count_nonzero(t & p)); fp = int(np.count_nonzero(~t & p))
    fn = int(np.count_nonzero(t & ~p)); tn = int(np.count_nonzero(~t & ~p))
    # IoU/Dice = 1 pour deux masques vides. Aucun tel cas dans le corpus utilisé.
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                fpr=fp/(fp+tn) if fp+tn else 0.)


CONDITIONS = ['original', 'dark', 'bright', 'warm', 'noise', 'jpeg']


def perturb(rgb, condition, seed):
    """Transformations photométriques seules : le masque reste aligné."""
    x = rgb.astype(np.float32)/255.
    if condition == 'original': return rgb.copy()
    if condition == 'dark': x = x**1.8
    elif condition == 'bright': x = x**.6
    elif condition == 'warm': x = x*np.array([1.2, 1., .8], np.float32)
    elif condition == 'noise':
        x = x + np.random.default_rng(seed).normal(0, .04, x.shape)
    elif condition == 'jpeg':
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 35])
        if not ok: raise RuntimeError('Encodage JPEG impossible')
        return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else: raise ValueError(condition)
    return np.clip(x*255., 0, 255).astype(np.uint8)
