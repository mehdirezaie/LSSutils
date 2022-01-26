'''
    Test the healpix based pair-counter
    Sep 25, 2019

    (c) Mehdi Rezaie

    > python test.py data/weight_comp_False_scaled_False_Nside_256_LRGnew.txtfc.fits data/weight_comp_False_scaled_True_Nside_256_LRG_randnew.txtfc.fits data/galDESILRGFalse2562ptPixclb.dat
    
'''


import healpy as hp
import numpy  as np
import sys
from time import time
import ddthetahp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

# read data
galaxy = hp.read_map(sys.argv[1])
random = hp.read_map(sys.argv[2])
ashley = np.loadtxt(sys.argv[3])

# set up over-density
delta  = np.zeros_like(galaxy)
mask   = random >= 0.2
sf     = galaxy[mask].sum()/random[mask].sum()
delta[mask] = galaxy[mask]/(random[mask]*sf) - 1.0

delta  = delta[mask]
random = random[mask]


#
#
# set up the bins
nside = 256
theta1, phi1 = hp.pix2ang(nside, np.argwhere(mask).flatten())
minang  = 0.15*np.pi/180.
maxang  = 12.0*np.pi/180.
binsize = 0.15*np.pi/180.
angles  = np.arange(minang, maxang, binsize)
bins    = np.cos(angles)[::-1]




bins_mid = 0.5*(bins[1:] + bins[:-1])

t0 = time()
c = ddthetahp.ddthetahpauto(theta1, phi1, delta, random, bins)
t1 = time()
print('time : ', t1-t0)



plt.plot(ashley[:, 0], ashley[:,1], marker='o', label='Ashley')
plt.plot(np.arccos(bins_mid)*180./np.pi, c[0,:]/c[1,:], 
            marker='x', alpha=0.5, label='Mehdi')

plt.legend()
plt.savefig('xi_test.png', bbox_inches='tight')
