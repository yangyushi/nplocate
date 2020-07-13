import numpy as np
import trackpy as tp
import nplocate as nl


diameter = 11
img = np.load('image.npy')
xyz = tp.locate(img, diameter=diameter)
xyz = np.array(xyz)[:, :3]

should_add = True
while should_add:
    n0 = xyz.shape[0]
    xyz = nl.add(xyz, img, diameter * 2, diameter, lambda im : np.array(tp.locate(im, diameter))[:, :3])
    n1 = xyz.shape[0]
    should_add = n1 > n0
xyz = nl.refine(xyz, img, diameter * 2, diameter)
