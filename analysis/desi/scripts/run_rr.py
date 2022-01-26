from time import time
import numpy as np
import healpy as hp
from lssutils.io import read_window
from lssutils.clustering import ddthetahpauto

nside = 512 #256
weight, mask = read_window('bmzls', nside)
theta, phi = hp.pix2ang(nside, np.argwhere(mask).flatten())

angles  = np.arange(0.0, np.pi, 0.01)
bins    = np.cos(angles)[::-1] # has to be increasing

t0 = time()
c = ddthetahpauto(theta, phi, 1.0*mask[mask], mask[mask]*1.0, bins)
t1 = time()
print('time : ', t1-t0)
np.save(f'bmzls_rr_window_{nside}', [angles[::-1], c[0]])
