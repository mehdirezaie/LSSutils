import numpy as np
import sys
sys.path.insert(0, '/home/mehdi/github/LSSutils')
from lssutils.stats import nnbar
from lssutils.utils import EbossCat
import healpy as hp





from lssutils.extrn.GalacticForegrounds.hpmaps import Gaia
ns = Gaia(nside_out=512)
nstarg = ns.nstar


def get_ngvsnstar(dat_fn, raw=2):
    dat = EbossCat(dat_fn, kind='data', zmin=0.8, zmax=2.2)
    ran = EbossCat(dat_fn.replace('.dat.', '.ran.'), kind='randoms', zmin=0.8, zmax=2.2)
    
    hpdat = dat.to_hp(512, 0.8, 2.2, raw=raw)
    hpran = ran.to_hp(512, 0.8, 2.2, raw=2)
    mask = (hpran > 0) & (np.isfinite(nstarg)) # & (np.isfinite(nstarg))    
    
    ones = np.ones_like(hpran)
    frac = hpran / (hp.nside2pixarea(512, degrees=True)*5000.)
    
    nbar_1 = nnbar.MeanDensity(hpdat, frac, mask, nstarg)
    nbar_1.run()
    return nbar_1.output


p0 = '/home/mehdi/data/eboss/data/v7_2/'
p = '/home/mehdi/data/eboss/data/v7_2/1.0/catalogs/'


d = {}
d['noweight'] = get_ngvsnstar(f'{p0}eBOSS_QSO_full_NGC_v7_2.dat.fits', raw=1)
d['NN-known'] = get_ngvsnstar(f'{p}eBOSS_QSO_full_NGC_known_mainhighz_512_v7_2.dat.fits')
d['NN-known+sdss'] = get_ngvsnstar(f'{p}eBOSS_QSO_full_NGC_known_mainstar_512_v7_2.dat.fits')
d['NN-known+gaia'] = get_ngvsnstar(f'{p}eBOSS_QSO_full_NGC_known_mainstarg_512_v7_2.dat.fits')
d['NN-all'] = get_ngvsnstar(f'{p}eBOSS_QSO_full_NGC_all_mainhighz_512_v7_2.dat.fits')
d['standard'] = get_ngvsnstar(f'{p0}eBOSS_QSO_full_NGC_v7_2.dat.fits')

np.save('/home/mehdi/data/eboss/data/v7_2/3.0/measurements/nnbar/nnbar_NGC_main_512_nstar', d)