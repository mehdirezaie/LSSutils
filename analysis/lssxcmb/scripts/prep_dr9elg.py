"""
    Prep tabulated data for DR9 ELGs
"""
import numpy as np
import fitsio as ft


path2ngal = '/fs/ess/PHS0336/data/tanveer/dr9/v6/dr9_elgs.npy'
regions = ['bmzls', 'ndecals', 'sdecals']

ng = np.load(path2ngal)
for region in regions:
    field = 'north' if region=='bmzls' else 'south'
    print(field, region)
    dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v4/nelg_features_{region}_clean_1024.fits')
    dt['label'] = ng[dt['hpix']]
    ft.write(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v6/nelg_features_{region}_1024.fits', dt)
