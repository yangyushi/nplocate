#!/usr/bin/env python
import sys
sys.path.insert(0, '..')
import matplotlib
matplotlib.use('TkAgg')
import nplocate as nl
from scipy import ndimage
import numpy as np
import trackpy as tp
import matplotlib.pyplot as plt


img = np.zeros((100, 100, 100))
img[50, 50, 50] = 1
img = ndimage.maximum_filter(img, size=5)
img = ndimage.gaussian_filter(img, (2, 2, 5))


m = nl.GaussianSphere(3, 2, 5)
pos = np.array((50, 50, 50)).astype(float).reshape((1, 3))
m.plot_compare(img, pos)

