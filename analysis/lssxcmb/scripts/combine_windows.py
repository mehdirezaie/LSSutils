"""
    Combine windows into HEALPix
"""
import sys
from glob import glob
import numpy as np
import fitsio as ft
import healpy as hp
sys.path.append('/home/mehdi/github/LSSutils')
import lssutils.utils as ut


def read_wnn(path2file):
    w_ = ft.read(path2file)
    w_nn = w_['weight'] / np.median(w_['weight'])
    w_nn = w_nn.clip(0.5, 2.0)
    return ut.make_hp(1024, w_['hpix'], w_nn, True)


def combine(wnn_list, output_map):
    wnn = []
    for wnn_i in wnn_list:
        wnn.append(read_wnn(wnn_i))
        print('.', end='')
    wnn_m = np.array(wnn).mean(axis=0)
    hp.write_map(output_map, wnn_m, fits_IDL=False)
    print(f'wrote {output_map}')

region = sys.argv[1]
wnn_list = glob(f'/home/mehdi/data/tanveer/dr9/elg_mse_snapshots/{region}/windows/wind*fits')
output_map = f'/home/mehdi/data/tanveer/dr9/elg_mse_snapshots/{region}/windows_mean.hp1024.fits'

combine(wnn_list, output_map)
