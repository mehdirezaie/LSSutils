"""
    Perform linear regression
    
"""
import os
import sys
import fitsio as ft
import numpy as np
import zeus
import emcee
from scipy.optimize import curve_fit

eps = 1.0e-8
np.random.seed(85)


def model(x, *theta):
    """ Linear model """
    return x.dot(theta[1:]) + theta[0]

def logprior(theta):
    ''' The natural logarithm of the prior probability. '''
    lp = 0.

    ## Gaussian prior on ?
    mmu = 0.0     # mean of the Gaussian prior
    msigma = 1.0 # standard deviation of the Gaussian prior
    
    #for theta_i in theta:
    #    lp -= 0.5*((theta_i - mmu)/msigma)**2

    lp = -0.5*((np.array(theta)-mmu)/msigma)**2
    lp = lp.sum()
    return lp

def loglike(theta, y, x):
    '''The natural logarithm of the likelihood.'''
    md = model(x, *theta)
    res = y-md
    err = y+eps  # Poisson error, pick some
    return -0.5 * (res*res/err).sum()

def logpost(theta, y, x):
    '''The natural logarithm of the posterior.'''
    return logprior(theta) + loglike(theta, y, x)

region = sys.argv[1]
ndim = 14     # Number of parameters/dimensions (e.g. m and c)
nwalkers = 400 # 400-500 walkers? Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 1000 # Number of steps/iterations.

#data_path = '/home/mehdi/data/tanveer/dr8/dr8_elg_ccd_1024_sub.fits'
data_path = f'/home/mehdi/data/rongpu/imaging_sys/tables/nelg_features_{region}_1024.fits'
output_path = f'/home/mehdi/data/tanveer/dr9/elg_linear/mcmc_{region}_poissonerr.npz'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'chains will be written on {output_path}')



data = ft.read(data_path)
x = data['features']
y = data['label']
print(f'# of features: {x.shape}')
assert np.all((y+eps) > 0)
# sub-sample
#ix = np.random.choice(np.arange(y.size), size=sub_size, replace=False)
#x = x[ix]
#y = y[ix]
#




# normalize the features
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

sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=[ys, xs]) # Initialise the sampler
sampler.run_mcmc(start, nsteps, progress=True) # Run sampling
#print(sampler.summary) # Print summary diagnostics
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
#print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))

chain = sampler.get_chain()
np.savez(output_path, **{'chain':chain, 'x':(xmean, xstd), 'y':(ymean, ystd)})
