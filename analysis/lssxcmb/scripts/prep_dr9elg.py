"""
    Prep tabulated data for DR9 ELGs
"""
import numpy as np
import fitsio as ft

ng = np.load('/fs/ess/PHS0336/data/tanveer/dr9/dr9_map/map_tomo.npy')
for region in ['bmzls', 'ndecals', 'sdecals']:
    field = 'north' if region=='bmzls' else 'south'
    print(field, region)

    dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v4/nelg_features_{region}_clean_1024.fits')
    dt['label'] = ng[dt['hpix']]
     
    ft.write(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v5/nelg_features_{region}_1024.fits', dt)
