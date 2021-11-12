import numpy as np
import os
import sys
sys.path.append('/home/mehdi/github/LSSutils')
import fitsio as ft
from lssutils.utils import EnsembleWeights
from astropy.table import Table
from glob import glob


nside = 256
cat_path = sys.argv[1] #'NBMZLS'
target = sys.argv[2] #'LRG'
mversion = 'v0'

root_dir = '/home/mehdi/data/dr9v0.57.0/'
wsys_dir = f'{root_dir}sv3nn_v1/regression/{mversion}/'
weights_ = glob(f'{wsys_dir}sv3nn_{target}_*_{nside}/nn-weights.fits')
weights = []
for w_ in weights_:
    weights.append(ft.read(w_))
weights = np.concatenate(weights)


out_path = f'{cat_path}_MrWsys/wsys_{mversion}.fits'
out_dir = os.path.dirname(out_path)

print(f'read raw weights from {weights_}')
if not os.path.exists(out_dir):
    print(f'create {out_dir}')
    os.makedirs(out_dir)
print(f'write output weights in {out_path}')


dcat = ft.read(cat_path, columns=['RA', 'DEC'])
ew = EnsembleWeights(weights, nside, istable=True)
w_ = ew(dcat['RA'], dcat['DEC'])

assert np.all(np.isfinite(w_) & (w_ > 0))

table = Table(data=[w_], names=['wsys'])
table.write(out_path, format='fits', overwrite=True)
