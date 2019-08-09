
from glob import glob
import healpy as hp
import numpy as np
import sys


#filename = '/Volumes/TimeMachine/data/eboss/v6/mask.cut.hp.512.fits'
#weights  = glob('/Volumes/TimeMachine/data/eboss/v6/results/regression/*/*weights.hp512.fits')
#mask    = hp.read_map('/Volumes/TimeMachine/data/eboss/v6/mask.hp.512.fits', verbose=False).astype('bool')


maskn   = sys.argv[1]
filename  = sys.argv[2]
weights = sys.argv[3:]
mask = hp.read_map(maskn, verbose=False).astype('bool')

for n in weights:
    print('map : %s '%n, end=' ')
    wi = hp.read_map(n, verbose=False)
    maski = (wi > 0.5) & (wi < 2.0)
    print(mask.sum(), maski.sum())
    mask &= maski    

    
hp.write_map(filename, mask, overwrite=True, fits_IDL=False)


print('total number of weights %d'%len(weights))
print('plain footprint %d'%mask.sum())
print('saving %s'%filename)

'''
(py3p6) bash-3.2$ time bash analyze_v6.sh 
map : /Volumes/TimeMachine/data/eboss/v6/results/regression/mult_all/lin-weights.hp512.fits  392502 392490
map : /Volumes/TimeMachine/data/eboss/v6/results/regression/mult_all/quad-weights.hp512.fits  392490 392464
map : /Volumes/TimeMachine/data/eboss/v6/results/regression/nn_ab/nn-weights.hp512.fits  392457 392498
map : /Volumes/TimeMachine/data/eboss/v6/results/regression/nn_p/nn-weights.hp512.fits  392455 392489
total number of weights 4
plain footprint 392450
saving /Volumes/TimeMachine/data/eboss/v6/mask.cut.hp.512.fits

real    0m1.772s
user    0m2.233s
sys     0m0.420s
'''
