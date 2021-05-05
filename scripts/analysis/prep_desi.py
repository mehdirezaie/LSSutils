import sys
import os
import fitsio as ft
import numpy as np
import pandas as pd
from time import time
from lssutils.lab import hpixsum, to_numpy

region = sys.argv[1] # NDECALS
target = sys.argv[2] # QSO

assert region in ['NDECALS', 'SDECALS', 'NBMZLS']
assert target in ['QSO', 'LRG', 'ELG', 'BGS_ANY']

nside = 256
root_path = '/home/mehdi/data/'
dat_path = f'{root_path}sv3/sv3target_{target}_{region}.fits'
ran_path = f'{root_path}sv3/{region}_randoms-1-0x5.fits'
tem_path = f'{root_path}templates/dr9/dr9m_nside{nside}.h5'
tab_path = f'{root_path}sv3nn/tables/sv3tab_{target}_{region}_{nside}.fits'


tab_dir = os.path.dirname(tab_path)
if not os.path.exists(tab_dir):
    print(f'Creating {tab_dir}')
    os.makedirs(tab_dir)
print(f'Output table will be written in {tab_dir}')


t0 = time()
dat = ft.read(dat_path, columns=['RA', 'DEC'])
ran = ft.read(ran_path, columns=['RA', 'DEC'])
t1 = time()
print(f'Read the input catalogs in {t1-t0:.1f} secs')

dathp = hpixsum(nside, dat['RA'], dat['DEC'])*1.0
ranhp = hpixsum(nside, ran['RA'], ran['DEC'])*1.0
t2 = time()
print(f'Project data and randoms to hp in {t2-t1:.1f} secs')

mask = ranhp > 0.0
frac = np.zeros_like(ranhp)
frac[mask] = ranhp[mask] / ranhp[mask].mean()
t3 = time()
print(f'Compute pixel completeness in {t3-t2:.1f} secs')

tmpl = pd.read_hdf(tem_path)
t4 = time()
print(f'Read imaging maps in {t4-t3:.1f} secs')

columns = ['nstar', 'ebv', 'loghi']\
          +[f'{s}_{b}' for s in ['ccdskymag_mean', 'fwhm_mean', 'fwhm_min', 'fwhm_max', 'depth_total', 
                                'mjd_mean', 'mjd_min', 'mjd_max', 'airmass_mean', 'exptime_total']\
                      for b in ['g', 'r', 'z']]

tmpl_np = tmpl[columns].values

mask_t = mask.copy()
for i in range(tmpl_np.shape[1]):
    mask_t &= np.isfinite(tmpl_np[:, i])
print(f'before: {mask.sum()}, after: {mask_t.sum()}')

d = to_numpy(dathp[mask_t], tmpl_np[mask_t], 
             frac[mask_t], np.argwhere(mask_t).flatten())
ft.write(tab_path, d)
print(f'{tab_path} is written!')
