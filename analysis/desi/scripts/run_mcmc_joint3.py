"""
    MCMC of Angular Power Spectrum

"""
import sys
import os
import emcee
import numpy as np

from multiprocessing import cpu_count, Pool

from scipy.optimize import minimize
from time import time
from lssutils.theory.cell import dNdz_model, init_sample, SurveySpectrum
from lssutils.extrn.mcmc import Posterior


SEED = 85

def read_inputs(path_cl, path_cov):
    
    dcl_obs = np.load(path_cl)
    dclcov_obs = np.load(path_cov)
    assert np.array_equal(dcl_obs['el_edges'], dclcov_obs['el_edges'])
    
    el_edges = dcl_obs['el_edges']
    cl_obs = dcl_obs['cl']
    invcov_obs = np.linalg.inv(dclcov_obs['clcov'])

    return el_edges, cl_obs, invcov_obs

def read_mask(region):

    import fitsio as ft
    import healpy as hp
    from lssutils.utils import make_hp

    if region in ['bmzls', 'ndecals', 'sdecals', 'ngc', 'desi']:
        # read survey geometry
        data_path = '/fs/ess/PHS0336/data/'    
        dt = ft.read(f'{data_path}/rongpu/imaging_sys/tables/0.57.0/nlrg_features_{region}_256.fits')
        mask_ = make_hp(256, dt['hpix'], 1.0) > 0.5
        mask = hp.ud_grade(mask_, 1024)
    else:
        # full sky
        mask = np.ones(12*1024*1024, '?')

    return mask*1.0, mask

    

# --- inputs
path_cl  = sys.argv[1:4]
path_cov = sys.argv[4:7]
region   = sys.argv[7:10]  # for window
output   = sys.argv[10]

print(path_cl)
print(path_cov)
print(region)



nsteps   = 10000   # int(sys.argv[2])
ndim     = 7      # Number of parameters/dimensions
nwalkers = 50     # Number of walkers to use. It should be at least twice the number of dimensions.
assert nwalkers > 2*ndim

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))    

if not os.path.exists(os.path.dirname(output)):
    print(f'create {os.path.dirname(output)}')
    os.makedirs(os.path.dirname(output))

z, b, dNdz = init_sample(kind='lrg')

# region 1
el_edges, cl_obs, invcov_obs = read_inputs(path_cl[0], path_cov[0])
weight, mask = read_mask(region[0])

model = SurveySpectrum()
model.add_tracer(z, b, dNdz, p=1.0)
model.add_kernels(model.el_model)
model.add_window(weight, mask, np.arange(2048), ngauss=2048)  

lg = Posterior(model, cl_obs, invcov_obs, el_edges)

# region 2
el_edges2, cl_obs2, invcov_obs2 = read_inputs(path_cl[1], path_cov[1])
weight, mask = read_mask(region[1])

model2 = SurveySpectrum()
model2.add_tracer(z, b, dNdz, p=1.0)
model2.add_kernels(model2.el_model)
model2.add_window(weight, mask, np.arange(2048), ngauss=2048)  

lg2 = Posterior(model2, cl_obs2, invcov_obs2, el_edges2)

# region 3
el_edges3, cl_obs3, invcov_obs3 = read_inputs(path_cl[2], path_cov[2])
weight, mask = read_mask(region[1])

model3 = SurveySpectrum()
model3.add_tracer(z, b, dNdz, p=1.0)
model3.add_kernels(model3.el_model)
model3.add_window(weight, mask, np.arange(2048), ngauss=2048)  

lg3 = Posterior(model3, cl_obs3, invcov_obs3, el_edges3)

def logpost(params):
    fnl, b1, sn1, b2, sn2, b3, sn3 = params
    return lg.logpost([fnl, b1, sn1]) + lg2.logpost([fnl, b2, sn2]) + lg3.logpost([fnl, b3, sn3])

def neglogpost(params):
    return -1*logpost(params)

# scipy optimization        
np.random.seed(SEED)
res = minimize(neglogpost, [1.0, 1.0, 1.0e-7, 1.0, 1.0e-7, 1.0, 1.0e-7], method='Powell')

# Initial positions of the walkers.
start = res.x *(1.+0.01*np.random.randn(nwalkers, ndim))
print(f'scipy opt: {res}')
print(f'initial guess: {start[:2]} ... {start[-1]}')

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool)
    sampler.run_mcmc(start, nsteps, progress=True)

np.savez(output, **{'chain':sampler.get_chain(), 
                    'log_prob':sampler.get_log_prob(), 
                    'best_fit':res.x,
                    'best_fit_logprob':res.fun,
                    'best_fit_success':res.success, 
                    '#data':cl_obs.size+cl_obs2.size,
                    '#params':ndim})
