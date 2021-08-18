"""
    Create density contrast and fracarea maps in HEALPix
    NSIDE=1024
    
"""

import fitsio as ft
import sys
sys.path.append('/home/mehdi/github/LSSutils')
from lssutils.lab import make_overdensity
from lssutils.utils import make_hp
import healpy as hp


def d2dl(d):
    
    nside = 1024
    ng = make_hp(nside, d['hpix'], d['label'])
    fr = make_hp(nside, d['hpix'], d['fracgood'])
    ms = fr > 0.0

    dl = make_overdensity(ng, fr, ms)
    
    return fr, dl


for r in ['bmzls', 'ndecals', 'sdecals']:
    
    d = ft.read(f'/home/mehdi/data/rongpu/imaging_sys/tables/nelg_features_{r}_1024.fits')
    
    fr, dl = d2dl(d)
    hp.write_map(f'/home/mehdi/data/tanveer/dr9/delta_elg_{r}_1024.hp.fits', dl, fits_IDL=False)
    hp.write_map(f'/home/mehdi/data/tanveer/dr9/fracarea_elg_{r}_1024.hp.fits', fr, fits_IDL=False)