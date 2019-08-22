# 
import Corrfunc
import Corrfunc.theory.DDsmu as DDsmu
import numpy as np
import sys
import os

from time import time

filename_input  = sys.argv[1]
filename_output = sys.argv[2]

# Corrfunc parameters
nthreads = int(sys.argv[3])

# read
if not os.path.isfile(filename_input):raise RuntimeError('$s not exists'%filename)

#data = Corrfunc.io.read_ascii_catalog(filename)
X1, Y1, Z1 = np.loadtxt(filename_input, usecols=(0, 1, 2), unpack=True)


ind    = np.random.choice(np.arange(0, X1.size), size=200000, replace=False)
X1 = X1[ind]
Y1 = Y1[ind]
Z1 = Z1[ind]

#
# DD(s,mu)
#
# prepare the inputs
autocorr = 1 # if 1, the second set of positions not required
nbins    = 100
binfile  = np.linspace(0., 200, nbins+1)
mu_max   = 1.0
nmu_bins = 120
boxsize  = 1000.  # Mpc/h




t0 = time()
results = DDsmu(autocorr, nthreads, binfile, mu_max, nmu_bins, X1, Y1, Z1, 
                periodic=True, verbose=True, boxsize=boxsize)
t1 = time()

np.save(filename_output, results)
print('took {} sec with {} threads'.format(t1-t0, nthreads))
print('saved the output on {}'.format(filename_output))
