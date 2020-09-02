""" Swap WEIGHT_SYSTOT with NN weights """
import os
from argparse import ArgumentParser

import lssutils.utils as ut
from lssutils import setup_logging
setup_logging('info')

# /users/PHS0336/medirz90/data/v7/1.0/0001/0/512/main/nn_pnnl_known
def nnw_path(path, idmock, iscont, nside, sample, maps):
    return os.path.join(path, f'{idmock}/{iscont}/{nside}/{sample}/nn_pnnl_{maps}/nn-weights.fits')

def prepare_mappers(path, idmock, iscont, nside, samples, maps, z_bins):
    """ prepare mappers for NN weights """
    mappers = {}
    
    for sample in samples:
    
        if sample not in z_bins:
            continue
            
        nnw_file = nnw_path(path, idmock, iscont, nside, sample, maps)
        if os.path.exists(nnw_file):
            mappers[sample] = (z_bins[sample], ut.NNWeight(nnw_file, nside))
            
    return mappers



ap = ArgumentParser(description='Post NN regression processing')
ap.add_argument('-m','--maps', type=str, default='known')
ap.add_argument('--zmin', type=float, default=0.8)
ap.add_argument('--zmax', type=float, default=2.2)
ap.add_argument('-n', '--nside', type=int,   default=512)
ap.add_argument('-s', '--samples', type=str, default=['main'], nargs='*')
ap.add_argument('-c', '--cap', type=str,   default='NGC')
ap.add_argument('--idmock', type=str)
ap.add_argument('--iscont', type=int)
ns = ap.parse_args() 


cap = ns.cap
nside = ns.nside
maps = ns.maps
cat_kw = dict(zmin=ns.zmin, zmax=ns.zmax)
samples = ns.samples
idmock = ns.idmock
iscont = ns.iscont

if iscont == 1:
    path_incats =  '/users/PHS0336/medirz90/data/v7/catalogs_raw/contaminated/'
    incat = f'{path_incats}EZmock_eBOSS_QSO_{cap}_v7_{idmock}.dat.fits'
    inran = incat.replace('.dat.', '.ran.')
else:
    path_incats =  '/users/PHS0336/medirz90/data/v7/catalogs_raw/null/'
    incat = f'{path_incats}EZmock_eBOSS_QSO_{cap}_v7_noweight_{idmock}.dat.fits'
    inran = incat.replace('.dat.', '.ran.')
 

path_weights = '/users/PHS0336/medirz90/data/v7/1.0'

samples_joined = ''.join(samples)
dat_name = os.path.join(path_weights, 'catalogs', f'EZmock_eBOSS_QSO_{cap}_{maps}_{samples_joined}_{nside}_v7_{iscont}_{idmock}.dat.fits')
ran_name = dat_name.replace('.dat.', '.ran.')
dat_dir = os.path.dirname(dat_name)
if not os.path.exists(dat_dir):
    os.makedirs(dat_dir)




# read data, randoms, and prepare mappers
dat = ut.EbossCat(incat, **cat_kw)
ran = ut.EbossCat(inran, kind='randoms', **cat_kw)
mappers = prepare_mappers(path_weights, idmock, iscont, nside, samples, maps, ut.z_bins)


# swap weight_systot weights, and reassign z-attrs to randoms
dat.swap(mappers)
ran.reassign_zattrs(dat)

# write to fits
dat.to_fits(dat_name)
ran.to_fits(ran_name)

print('wrote', dat_name)
print('wrote', ran_name)

