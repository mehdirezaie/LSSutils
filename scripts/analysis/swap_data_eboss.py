""" Swap WEIGHT_SYSTOT with NN weights """
import os
from argparse import ArgumentParser

import lssutils.utils as ut
from lssutils import setup_logging
setup_logging('info')


def nnw_path(path, cap, nside, sample, maps, method):
    return os.path.join(path, f'{cap}/{nside}/{sample}/{method}_{maps}/nn-weights.fits')

def prepare_mappers(path, cap, nside, samples, maps, z_bins, method):
    """ prepare mappers for NN weights """
    mappers = {}
    
    for sample in samples:
    
        if sample not in z_bins:
            print(f'{sample} does not exist')
            continue
            
        nnw_file = nnw_path(path, cap, nside, sample, maps, method)
        if os.path.exists(nnw_file):
            print(f'{nnw_file} does exist')
            mappers[sample] = (z_bins[sample], ut.NNWeight(nnw_file, nside))
        else:
            print(f'{nnw_file} does not exist')
    return mappers



ap = ArgumentParser(description='Post NN regression processing')
ap.add_argument('-m','--maps', type=str, default='all')
ap.add_argument('--zmin', type=float, default=0.8)
ap.add_argument('--zmax', type=float, default=3.5)
ap.add_argument('-n', '--nside', type=int,   default=512)
ap.add_argument('-s', '--samples', type=str, default=['main', 'highz'], nargs='*')
ap.add_argument('-c', '--cap', type=str,   default='NGC')
ap.add_argument('-v', '--version', type=str,   default='1.0')
ap.add_argument('--method', type=str, default='nn_pnll')
ns = ap.parse_args() 

cap = ns.cap
nside = ns.nside
maps = ns.maps
cat_kw = dict(zmin=ns.zmin, zmax=ns.zmax)
samples = ns.samples
method = ns.method
version = ns.version

path_incats =  '/home/mehdi/data/eboss/data/v7_2/'
path_weights = f'/home/mehdi/data/eboss/data/v7_2/{version}/'

samples_joined = ''.join(samples)
dat_name = os.path.join(path_weights, 'catalogs', f'eBOSS_QSO_full_{cap}_{maps}_{samples_joined}_{nside}_v7_2.dat.fits')
ran_name = dat_name.replace('.dat.', '.ran.')
dat_dir = os.path.dirname(dat_name)
if not os.path.exists(dat_dir):
    os.makedirs(dat_dir)

if os.path.exists(dat_name):
    raise RuntimeError(f'{dat_name} exists')

if os.path.exists(ran_name):
    raise RuntimeError(f'{ran_name} exists')



# read data, randoms, and prepare mappers
dat = ut.EbossCat(f'{path_incats}eBOSS_QSO_full_{cap}_v7_2.dat.fits', **cat_kw)
ran = ut.EbossCat(f'{path_incats}eBOSS_QSO_full_{cap}_v7_2.ran.fits', kind='randoms', **cat_kw)
mappers = prepare_mappers(path_weights, cap, nside, samples, maps, ut.z_bins, method)

# swap weight_systot weights, and reassign z-attrs to randoms
dat.swap(mappers)
ran.reassign_zattrs(dat)

# write to fits
dat.to_fits(dat_name)
ran.to_fits(ran_name)

print('wrote', dat_name)
print('wrote', ran_name)

