import os
import numpy as np
import healpy as hp
from glob import glob
import fitsio as ft

windows = dict()
for region in ['bmzls', 'ndecals', 'sdecals']:
    windows[region] = glob(f'/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_dnn/{region}_1024/windows/window_model_*fits')
    print(region)


nwindows = 1000
nside = 1024
for i in range(nwindows):
    
    wind_i = np.zeros(12*nside*nside)
    for region in windows:
        d_ = ft.read(windows[region][i])
        wind_i[d_['hpix']] += d_['weight'] 

    output_path = f'/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_dnn/windows/nnwindow_{i}.hp{nside}.fits'
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
         os.makedirs(output_dir)
    print(f'wrote {output_path}')
    hp.write_map(output_path, wind_i, dtype=np.float64)    
