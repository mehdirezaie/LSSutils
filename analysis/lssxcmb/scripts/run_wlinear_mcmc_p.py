"""
    Perform linear regression
    
"""
import os
import sys
import fitsio as ft
import numpy as np
import emcee
from scipy.optimize import curve_fit
from multiprocessing import cpu_count, Pool

eps = 1.0e-8
np.random.seed(85)


def modelp(x, *theta):
    """ Linear model Poisson """
    return np.log(1.+np.exp(x.dot(theta[1:]) + theta[0]))

def logprior(theta):
    ''' The natural logarithm of the prior probability. '''
    ## Gaussian prior on ?
    mmu = 0.0     # mean of the Gaussian prior
    msigma = 1.0 # standard deviation of the Gaussian prior
    
    lp = 0. if 0.0 < theta[0] < 1000. else -np.inf
    lp += -0.5*((np.array(theta[1:])-mmu)/msigma)**2
    lp = lp.sum()
    return lp

def loglike(theta, y, x, w):
    '''The natural logarithm of the (Gaussian) likelihood.'''
    md = modelp(x, *theta)
    res = (w*md-y*np.log(w*md))
    return -0.5 * (w*res).sum()


from argparse import ArgumentParser
ap = ArgumentParser(description='Linear Regression')
ap.add_argument('-d', '--data_path', required=True)
ap.add_argument('-o', '--output_path', required=True)
ap.add_argument('-ax', '--axes', nargs='*', type=int) 
ns = ap.parse_args()

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))    

data_path = ns.data_path   
output_path = ns.output_path

ndim = len(ns.axes)+1     # Number of parameters/dimensions (e.g. m and c) +1 for bias
nwalkers = 400            # 400-500 walkers? Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 1000             # 1000, Number of steps/iterations.
print(f'axes for regression: {ns.axes}')

output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'chains will be written on {output_path}')


data = ft.read(data_path)
x = data['features'][:, ns.axes]
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

def logpost(theta):
    '''The natural logarithm of the posterior.'''
    return logprior(theta) + loglike(theta, y, xs, w)

#ymean = np.mean(y)
#ystd = np.std(y)
#ys = (y - ymean)/ystd

p0 = np.zeros(ndim)
results = curve_fit(modelp, xs, y, p0=p0)
print('curve_fit:', results[0])

start = results[0] + 0.1*np.random.randn(nwalkers, ndim) # Initial positions of the walkers.
print(f'initial guess: {start[0]}')

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool) # Initialise the sampler
    sampler.run_mcmc(start, nsteps, progress=True) # Run sampling
    #print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

np.savez(output_path, **{'chain':sampler.get_chain(), 'x':(xmean, xstd)}) #, 'y':(ymean, ystd)})
