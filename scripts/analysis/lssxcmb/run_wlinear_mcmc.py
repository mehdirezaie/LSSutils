"""
    Perform linear regression
    
"""
import os
import sys
import fitsio as ft
import numpy as np
import emcee
from scipy.optimize import curve_fit

eps = 1.0e-8
np.random.seed(85)


def model(x, *theta):
    """ Linear model """
    return x.dot(theta[1:]) + theta[0]

def logprior(theta):
    ''' The natural logarithm of the prior probability. '''
    ## Gaussian prior on ?
    mmu = 0.0     # mean of the Gaussian prior
    msigma = 1.0 # standard deviation of the Gaussian prior
    
    lp = -0.5*((np.array(theta)-mmu)/msigma)**2
    lp = lp.sum()
    return lp

def loglike(theta, y, x, w):
    '''The natural logarithm of the (Gaussian) likelihood.'''
    md = model(x, *theta)
    res = (y-w*md)
    return -0.5 * (w*res*res).sum()

def logpost(theta, y, x, w):
    '''The natural logarithm of the posterior.'''
    return logprior(theta) + loglike(theta, y, x, w)

data_path = sys.argv[1]   # '/home/mehdi/data/rongpu/imaging_sys/tables/nelg_features_{region}_1024.fits'
output_path = sys.argv[2] # f'/home/mehdi/data/tanveer/dr9/elg_linear/mcmc_{region}_poissonerr.npz'

ndim = 14                 # Number of parameters/dimensions (e.g. m and c) +1 for bias
nwalkers = 400            # 400-500 walkers? Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 1000             # 1000, Number of steps/iterations.

output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'chains will be written on {output_path}')


data = ft.read(data_path)
x = data['features']
y = data['label']
w = data['fracgood']
print(f'# of features: {x.shape}')
#assert np.all((y+eps) > 0)
# sub-sample
#ix = np.random.choice(np.arange(y.size), size=sub_size, replace=False)
#x = x[ix]
#y = y[ix]


# normalize the features and label
xmean = np.mean(x, axis=0)
xstd = np.std(x, axis=0)
xs = (x - xmean) / xstd

ymean = np.mean(y)
ystd = np.std(y)
ys = (y - ymean)/ystd

p0 = np.zeros(ndim)
results = curve_fit(model, xs, ys, p0=p0)
print('curve_fit:', results[0])

start = results[0] + 0.1*np.random.randn(nwalkers, ndim) # Initial positions of the walkers.
print(f'initial guess: {start[0]}')

sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=[ys, xs, w]) # Initialise the sampler
sampler.run_mcmc(start, nsteps, progress=True) # Run sampling
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

chain = sampler.get_chain()
np.savez(output_path, **{'chain':chain, 'x':(xmean, xstd), 'y':(ymean, ystd)})
