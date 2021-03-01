import sys
import numpy as np
from lssutils.stats import nnbar
from lssutils.utils import EbossCat, nside2pixarea
from lssutils.extrn.GalacticForegrounds.hpmaps import Gaia




dat_fn = sys.argv[1]
path_nnbar = sys.argv[2]
path_nnbar_new = sys.argv[3]

print(f'input: {dat_fn}')
print(f'nnbar in: {path_nnbar}')
print(f'nnbar out: {path_nnbar_new}')


path_gaia = '/fs/ess/PHS0336/data/templates/Gaia.dr2.bGT10.12g17.hp256.fits' 


dd = Gaia(path=path_gaia, nside_out=512)
nstarg = dd.nstar

dat = EbossCat(dat_fn, kind='data', zmin=0.8, zmax=2.2)
ran = EbossCat(dat_fn.replace('.dat.', '.ran.'), kind='randoms', zmin=0.8, zmax=2.2)

hpdat = dat.to_hp(512, 0.8, 2.2, raw=2)
hpran = ran.to_hp(512, 0.8, 2.2, raw=2)
mask = (hpran > 0) & (np.isfinite(nstarg)) # & (np.isfinite(nstarg))    

ones = np.ones_like(hpran)
frac = hpran / (nside2pixarea(512, degrees=True)*5000.)

nbar_1 = nnbar.MeanDensity(hpdat, frac, mask, nstarg)
nbar_1.run()


nnbar_old = np.load(path_nnbar, allow_pickle=True)
nnbar_oldc = nnbar_old.copy()
nnbar_oldc[0] = nbar_1.output

#print(nnbar_oldc[0])
np.save(path_nnbar_new, nnbar_oldc)
