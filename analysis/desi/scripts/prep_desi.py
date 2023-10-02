
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
tag_d = sys.argv[5]

assert tag_d in ['0.57.0', '1.0.0'], "data version not supported"
print(kind, region, nside, tag_d)

assert kind in ['lrg', 'elg', 'bgs_any', 'bgs_bright'], f'{kind} not implemented'
assert region in ['bmzls', 'ndecals', 'sdecals'], f'{region} not implemented'



maskbits = {'lrg':'_lrgmask_v1', #old:'lrg':189111213,
            'elg':1111213,
            'qso':np.nan,
            'bgs_any':113,
            'bgs_bright':113}

#tag_d = '0.57.0'
tag_r = '0.49.0'


cap = 'north' if region in ['bmzls'] else 'south'
path_out = os.path.join(path, 'tables', tag_d, f'n{kind}_features_{region}_{nside}.fits')
print(path_out)


dir_out = os.path.dirname(path_out)
if not os.path.exists(dir_out):
    print(f'{dir_out} does not exist')
    os.makedirs(dir_out)


mb = maskbits[kind]

if tag_d=='0.57.0':
    data_ng = ft.read(f'{path}/density_maps/{tag_d}/resolve/density_map_sv3_{kind}_{cap}_nside_{nside}_minobs_1_maskbits_{mb}.fits')
elif tag_d=='1.0.0':
    data_ng = ft.read(f'{path}/density_maps/{tag_d}/resolve/density_map_{kind}_{cap}_nside_{nside}_minobs_1_maskbits_{mb}.fits')

data_tmp = ft.read(f'{path}/randoms_stats/{tag_r}/resolve/combined/pixmap_{cap}_nside_{nside}_minobs_1_maskbits_{mb}.fits', upper=True)

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
    
#if region=='ndecals':
#    data_nn = ut.remove_islands(data_nn, nside)
    
ft.write(path_out, data_nn, clobber=True)



## ---  code below is for adding extra columns

"""
import fitsio as ft
import lssutils.utils as ut
from lssutils.stats.nnbar import get_meandensity
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from lssutils.extrn.galactic.hpmaps import logHI

def rotate(map1, mask1):
    map1[mask1] = np.nan
    map1_ = hp.reorder(map1, n2r=True)
    return map1_


def add_extras(reg):
    
    data = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_{reg}_256.fits')
    extra = ft.read('/fs/ess/PHS0336/data/templates/pixweight_external.fits')
    lh = logHI(path='/fs/ess/PHS0336/data/templates/NHI_HPX.fits')
    
    calibz = rotate(extra['CALIBZ'], extra['CALIBMASKZ']>0.5)
    #halpha = np.log10(rotate(extra['HALPHA'], extra['HALPHAMASK']==1))

    is_bad = np.isnan(calibz[data['hpix']])
    print('# bad', is_bad.sum())
    cmean = np.nanmean(calibz[data['hpix']])
    print(calibz[data['hpix']][is_bad])
    calibz[data['hpix'][is_bad]] = cmean
    print(cmean)
    print(calibz[data['hpix']][is_bad])
    sysm = np.column_stack([data['features'], calibz[data['hpix']], lh.map[data['hpix']]]) # halpha[data['hpix']]

    is_good = np.isnan(sysm).sum(axis=1) < 0.5
    assert(is_good.mean() ==1)

    data_new = ut.to_numpy(data['label'], sysm, data['fracgood'], data['hpix'])
    output = f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_{reg}ext_256.fits'
    ft.write(output, data_new)
    print('wrote ', output)
    
add_extras('sdecalsc')
"""