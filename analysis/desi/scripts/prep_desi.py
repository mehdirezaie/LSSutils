
import numpy as np
import fitsio as ft
from glob import glob
from scipy.stats import pearsonr
import sys
import os
import lssutils.utils as ut


kind = sys.argv[1]
region = sys.argv[2]
nside = int(sys.argv[3])
path = sys.argv[4] #'/fs/ess/PHS0336/data/rongpu/imaging_sys'
version = sys.argv[5]

print(kind, region, nside, version)

assert kind in ['lrg', 'elg', 'bgs_any', 'bgs_bright'], f'{kind} not implemented'
assert region in ['bmzls', 'ndecals', 'sdecals'], f'{region} not implemented'



maskbits = {'lrg':'_lrgmask_v1', #old:'lrg':189111213,
            'elg':1111213,
            'qso':np.nan,
            'bgs_any':113,
            'bgs_bright':113}

tag_d = '0.57.0'
tag_r = '0.49.0'


cap = 'north' if region in ['bmzls'] else 'south'
path_out = os.path.join(path, 'tables', version, f'n{kind}_features_{region}_{nside}.fits')
print(path_out)


dir_out = os.path.dirname(path_out)
if not os.path.exists(dir_out):
    print(f'{dir_out} does not exist')
    os.makedirs(dir_out)


mb = maskbits[kind]
data_ng = ft.read(f'{path}/density_maps/{tag_d}/resolve/density_map_sv3_{kind}_{cap}_nside_{nside}_minobs_1_maskbits_{mb}.fits')
data_tmp = ft.read(f'{path}/randoms_stats/{tag_r}/resolve/combined/pixmap_{cap}_nside_{nside}_minobs_1_maskbits_{mb}.fits')

# split south into sdecals and ndecals
if region in ['ndecals', 'sdecals']:
    is_region = ut.select_region(data_ng['RA'], data_ng['DEC'], region)
    data_ng = data_ng[is_region]
    
    is_region = ut.select_region(data_tmp['RA'], data_tmp['DEC'], region)
    data_tmp = data_tmp[is_region]
    

ngal = ut.make_hp(nside, data_ng['HPXPIXEL'], data_ng['n_targets'])
hpix = data_tmp['HPXPIXEL']
fracgood = data_tmp['FRACAREA']
label = ngal[hpix]
features = []
for col in ut.maps_dr9:
    feat_ = data_tmp[col]
    features.append(feat_)
features = np.array(features).T


data_nn = ut.to_numpy(label, features, fracgood, hpix)
for name in data_nn.dtype.names:
    assert (~np.isfinite(data_nn[name])).sum() == 0, f'{name} is bad'
ft.write(path_out, data_nn, clobber=True)
