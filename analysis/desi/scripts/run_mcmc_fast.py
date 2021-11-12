"""
    MCMC of Angular Power Spectrum

"""
import sys
import os
import emcee
import numpy as np

from multiprocessing import cpu_count, Pool

#from scipy.optimize import minimize
from time import time
from lssutils.theory.cell import dNdz_model, init_sample, SurveySpectrum
from lssutils.io import read_clmocks, read_window
from lssutils.extrn.mcmc import Posterior


region = sys.argv[1] #'bmzls'
method = sys.argv[2] #'noweight'
output = sys.argv[3] #'test_mcmc.npy'

if not os.path.exists(os.path.dirname(output)):
    print(f'create {os.path.dirname(output)}')
    os.makedirs(os.path.dirname(output))


ndim     = 2      # Number of parameters/dimensions
nwalkers = 10     # Number of walkers to use. It should be at least twice the number of dimensions.
nsteps   = 1000   # Number of steps/iterations.
print(region, method, output)

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

bins = np.array([2*(i+1) for i in range(10)]+[30+i*50 for i in range(7)])
print('bins: ', bins)


el, cl, invcov = read_clmocks(region, method, bins=bins)
weight, mask = read_window(region)

z, b, dNdz = init_sample(kind='lrg')

model = SurveySpectrum()
model.add_tracer(z, b, dNdz, p=1.6)
model.make_kernels(model.el_model)
model.prep_window(weight, mask, np.arange(2*1024), ngauss=2*1024)


#--- optimization
lg = Posterior(model, cl, invcov, el)
def logpost(foo):
    return lg.logpost(foo)

# ad hoc optimization
#for fnl in np.logspace(-3., 3., 5):
#    for noise in np.linspace(-1.0e-6, 1.0e-6, 4):
#        print(f'{fnl:5.1f}, {noise:10.4e}, {lg.logpost([fnl, noise], cl, invcov, el):10.4e}')


# scipy optimization        
#res = minimize(lg.logpost, [0., 1.0e-7], args=(cl, invcov, el), )
#print(res)

# Initial positions of the walkers. TODO: add res.x
start = np.array([10.0, 1.0e-5])*np.random.randn(nwalkers, ndim) 
print(f'initial guess: {start[:2]} ... {start[-1]}')

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool)
    sampler.run_mcmc(start, nsteps, progress=True)

np.save(output, sampler.get_chain(), allow_pickle=False)
