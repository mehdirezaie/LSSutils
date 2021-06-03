import numpy as np
import os
import sys
sys.path.append('/home/mehdi/github/LSSutils')
import fitsio as ft
from lssutils.utils import EnsembleWeights
from astropy.table import Table


nside = 256
region = sys.argv[2] #'NBMZLS'
target = sys.argv[1] #'LRG'
mversion = sys.argv[3] #'v0.1'

root_dir = '/home/mehdi/data/dr9v0.57.0/'
wsys_dir = f'/home/mehdi/data/dr9v0.57.0/sv3nn_v1/regression/{mversion}/'

region_ = 'SDECALS' if region in ['DES', 'SDECALS_noDES', 'DES_noLMC'] else region
wsys_path = f'{wsys_dir}sv3nn_{target}_{region_}_256/nn-weights.fits'

cats_dir = f'{root_dir}sv3_v1/'
cat_path = f'{cats_dir}sv3target_{target}_{region}.fits'

out_path = f'{cats_dir}sv3target_{target}_{region}.fits_MrWsys/wsys_{mversion}.fits'
out_dir = os.path.dirname(out_path)

print(f'read raw weights from {wsys_path}')
if not os.path.exists(out_dir):
    print(f'create {out_dir}')
    os.makedirs(out_dir)
print(f'write output weights in {out_path}')


dcat = ft.read(cat_path, columns=['RA', 'DEC'])
ew = EnsembleWeights(wsys_path, 256)
w_ = ew(dcat['RA'], dcat['DEC'])

assert np.all(np.isfinite(w_) & (w_ > 0))

table = Table(data=[w_], names=['wsys'])
table.write(out_path, format='fits', overwrite=True)
