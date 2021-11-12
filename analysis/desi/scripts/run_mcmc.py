"""
    MCMC of Angular Power Spectrum

"""
import sys
import emcee
#import zeus  ## TODO: 9x slower, why??
import fitsio as ft
import healpy as hp
import numpy as np

#from multiprocessing import cpu_count
#from multiprocessing.pool import ThreadPool as Pool

from glob import glob
from scipy.optimize import minimize
from time import time
from lssutils.theory.cell import (dNdz_model, init_sample,
                                  SurveySpectrum)
from lssutils.utils import histogram_cell, make_hp


def read_window(region):
    """ Return Window, Mask, Noise
    """
    t0 = time()
    # read survey geometry
    data_path = '/fs/ess/PHS0336/data/'    
    dt = ft.read(f'{data_path}/rongpu/imaging_sys/tables/v3/nlrg_features_{region}_256.fits')
    mask_  = make_hp(256, dt['hpix'], 1.0) > 0.5
    
    mask   = hp.ud_grade(mask_, 1024)
    weight = mask * 1.0
    noise  = 5.378248584004819e-07 # 1.25*1/np.mean(dt['label'])*hp.nside2pixarea(256)
    t1 = time()
    print(f'read window in {t1-t0:.2f} sec')
    
    return weight, mask, noise

def read_mocks(region, method, plot_cov=False):
    t0 = time()
    bins = np.array([2*(i+1) for i in range(10)] + [2**i for i in range(5, 9)])

    data_path = '/fs/ess/PHS0336/data/'
    list_clmocks = glob(f'{data_path}lognormal/v0/clustering/clmock_*_lrg_{region}_256_noweight.npy')
    print(len(list_clmocks))


    cl_list = []
    for cl_i in list_clmocks:
        cl_ = np.load(cl_i, allow_pickle=True).item()['cl_gg']['cl']
        lb_, clb_ = histogram_cell(cl_, bins=bins)        
        cl_list.append(clb_)
        print('.', end='')
    t1 = time()    
    print(f'\nfinished reading noweight mocks Cls in {t1-t0:.2f} sec')
    
    nmocks, nbins = np.array(cl_list).shape
    hf = (nmocks - 1.0)/(nmocks - nbins - 2.0)
    print(f'Hartlap with #mocks ({nmocks}) and #bins ({nbins})')
    cl_cov = np.cov(cl_list, rowvar=False)*hf / nmocks
    inv_cov = np.linalg.inv(cl_cov)
    
    if method in ['nn', 'lin']:
        list_clmocks = glob(f'{data_path}lognormal/v0/clustering/clmock_*_lrg_{region}_256_{method}.npy')
        print(len(list_clmocks))
        
        cl_list = []
        for cl_i in list_clmocks:
            cl_ = np.load(cl_i, allow_pickle=True).item()['cl_gg']['cl']
            lb_, clb_ = histogram_cell(cl_, bins=bins)        
            cl_list.append(clb_)
            print('.', end='')
        t1 = time()    
        print(f'\nfinished reading {method} mocks Cls in {t1-t0:.2f} sec')        
          
    cl_mean = np.mean(cl_list, axis=0)

#     if plot_cov:
#         vmin, vmax = np.percentile(cl_cov, [5, 95])
#         lim = np.minimum(abs(vmin), abs(vmax))        
#         plt.imshow(cl_cov, origin='lower', vmin=-1.*lim, vmax=lim, cmap=plt.cm.bwr)
#         plt.show()
#         plt.imshow(cl_cov.dot(inv_cov), origin='lower')
#         plt.show()
        
    print(cl_mean.shape, cl_cov.shape, len(cl_list))
    return (lb_.astype('int'), cl_mean, inv_cov)        


class Posterior:
    """ Log Posterior for PNGModel
    """
    def __init__(self, model):
        self.model = model

    def logprior(self, theta):
        ''' The natural logarithm of the prior probability. '''
        lp = 0.
        # unpack the model parameters from the tuple
        fnl, noise = theta
        
        # uniform prior on fNL
        fmin = -1.0e3 # lower range of prior
        fmax = 1.0e3  # upper range of prior
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp += 0. if fmin < fnl < fmax else -np.inf
        
        # uniform prior on noise
        noise_min = -1.0e-5
        noise_max =  1.0e-5
        lp += 0. if noise_min < noise < noise_max else -np.inf
        
        ## Gaussian prior on ?
        #mmu = 3.     # mean of the Gaussian prior
        #msigma = 10. # standard deviation of the Gaussian prior
        #lp += -0.5*((m - mmu)/msigma)**2

        return lp

    def loglike(self, theta, y, invcov, x):
        '''The natural logarithm of the likelihood.'''
        # unpack the model parameters
        fnl, noise = theta
        
        # evaluate the model
        md = self.model(x, fnl=fnl, noise=noise)
        # return the log likelihood
        return -0.5 * (y-md).dot(invcov.dot(y-md))

    def logpost(self, theta, y, invcov, x):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta, y, invcov, x)


region = sys.argv[1] #'bmzls'
method = sys.argv[2] #'noweight'
output = sys.argv[3] #'test_mcmc.npy'

print(region, method, output)



#ncpu = cpu_count()
#print("{0} CPUs".format(ncpu))

#--- read "data"
el, cl, invcov = read_mocks(region, method)  # read mean cl and inv cov of mocks
weight, mask, noise = read_window(region)

#--- initiate model
t0 = time()
z, b, dNdz = init_sample(kind='lrg')
t1 = time()
print(f'initiate sample dN/dz in {t1-t0:.2f} sec')

t0 = time()
model = SurveySpectrum()
model.add_tracer(z, b, dNdz, p=1.6)
model.make_kernels(model.el_model)
model.prep_window(weight, mask, np.arange(2*1024), ngauss=2*1024)
t1 = time()
print(f'initiate model in {t1-t0:.2f} sec')


#--- optimization
lg = Posterior(model)

# ad hoc optimization
#for fnl in np.logspace(-3., 3., 5):
#    for noise in np.linspace(-1.0e-6, 1.0e-6, 4):
#        print(f'{fnl:5.1f}, {noise:10.4e}, {lg.logpost([fnl, noise], cl, invcov, el):10.4e}')


# scipy optimization        
#res = minimize(lg.logpost, [0., 1.0e-7], args=(cl, invcov, el), )
#print(res)

# chains
ndim     = 2      # Number of parameters/dimensions
nwalkers = 10     # Number of walkers to use. It should be at least twice the number of dimensions.
nsteps   = 1000   # Number of steps/iterations.

start = np.array([1.0, 1.0e-7])*np.random.randn(nwalkers, ndim) # Initial positions of the walkers. TODO: add res.x
print(f'initial guess: {start[:2]} ... {start[-1]}')

sampler = emcee.EnsembleSampler(nwalkers, ndim, lg.logpost, 
                                args=(cl, invcov, el)) # Initialise the sampler
sampler.run_mcmc(start, nsteps, progress=True) # Run sampling

np.save(output, sampler.get_chain(), allow_pickle=False)
