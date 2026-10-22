import sys,unittest,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chromavision.core import *

class TestScientificCore(unittest.TestCase):
    def test_confusion_known(self):
        # TP=1, FP=1, FN=1, TN=1 : IoU=1/3 et Dice=1/2.
        m=binary_metrics(np.array([[1,1],[0,0]]),np.array([[1,0],[1,0]]))
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,1,1,1))
        self.assertAlmostEqual(m['iou'],1/3);self.assertAlmostEqual(m['dice'],.5)
    def test_rgb_subtraction_no_unsigned_overflow(self):
        x=np.array([[[180,100,50],[10,200,50]]],np.uint8)
        self.assertEqual(rgb_rule(x).tolist(),[[True,False]])
    def test_histogram_learns_color_not_position(self):
        im=np.array([[[190,130,90],[10,80,220]],[[190,130,90],[10,80,220]]],np.uint8)
        gt=np.array([[1,0],[1,0]],bool)
        model=fit_histogram([('tiny',im,gt)],'Hist-Lab',smooth=0,pixels_per_class=100)
        self.assertTrue(np.array_equal(model.predict(im),gt))
        self.assertTrue(np.array_equal(model.predict(im[:,::-1]),gt[:,::-1]))
        with tempfile.TemporaryDirectory() as t:
            model.save(t);loaded=ColorModel.load(t,model.name)
            np.testing.assert_array_equal(model.predict(im),loaded.predict(im))
    def test_bin_bounds_and_hue_wrap(self):
        rng=np.random.default_rng(2);im=rng.integers(0,256,(20,20,3),dtype=np.uint8)
        for space in ['Lab','HS','CrCb']:
            b=color_bins(im,space,16);self.assertGreaterEqual(b.min(),0);self.assertLess(b.max(),16)
    def test_perturbations_repeatable_no_geometry_change(self):
        im=np.full((12,15,3),128,np.uint8)
        for c in CONDITIONS:
            a=perturb(im,c,12);b=perturb(im,c,12)
            self.assertEqual(a.shape,im.shape);self.assertEqual(a.dtype,np.uint8)
            np.testing.assert_array_equal(a,b)
    def test_morphology_removes_isolated_false_positive(self):
        m=np.zeros((21,21),bool);m[5:16,5:16]=True;m[1,1]=True
        out=cleanup(m,3);self.assertFalse(out[1,1]);self.assertTrue(out[10,10])
    def test_size_mismatch(self):
        with self.assertRaises(ValueError):binary_metrics(np.zeros((2,2)),np.zeros((3,3)))

if __name__=='__main__':unittest.main()
