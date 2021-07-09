"""
    Perform linear regression
    
"""
import fitsio as ft
import numpy as np
import zeus
from scipy.optimize import curve_fit

def model(x, *theta):
    """ Linear model """
    return x.dot(theta[1:]) + theta[0]

def logprior(theta):
    ''' The natural logarithm of the prior probability. '''
    lp = 0.

    ## Gaussian prior on ?
    mmu = 0.0     # mean of the Gaussian prior
    msigma = 1.0 # standard deviation of the Gaussian prior
    
    for theta_i in theta:
        lp -= 0.5*((theta_i - mmu)/msigma)**2

    return lp

def loglike(theta, y, x):
    '''The natural logarithm of the likelihood.'''
    md = model(x, *theta)
    return -0.5 * ((y-md)**2).sum()

def logpost(theta, y, x):
    '''The natural logarithm of the posterior.'''
    return logprior(theta) + loglike(theta, y, x)



data_path = '/home/mehdi/data/tanveer/dr8_elg_ccd_1024_sub.fits'
#data_path = '/home/mehdi/data/tanveer/dr8_elg_ccd_1024.fits'

data = ft.read(data_path)
x = data['features']
y = data['label']

# normalize the features
xmean = np.mean(x, axis=0)
xstd = np.std(x, axis=0)
xs = (x - xmean) / xstd

ymean = np.mean(y)
ystd = np.std(y)
ys = (y - ymean)/ystd
np.savez('stats_xy.npz', **{'x':(xmean, xstd), 'y':(ymean, ystd)})


ndim = 22      # Number of parameters/dimensions (e.g. m and c)
nwalkers = 400 # 400-500 walkers? Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 1000 # Number of steps/iterations.

p0 = np.zeros(22)
results = curve_fit(model, xs, ys, p0=p0)
print('curve_fit:', results[0])

start = results[0] + 0.1*np.random.randn(nwalkers, ndim) # Initial positions of the walkers.
print(f'initial guess: {start[0]}')

sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost, args=[ys, xs]) # Initialise the sampler
sampler.run_mcmc(start, nsteps) # Run sampling
print(sampler.summary) # Print summary diagnostics

chain = sampler.get_chain()
np.save('chains.npy', chain)
