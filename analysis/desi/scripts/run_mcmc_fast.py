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

    if region in ['bmzls', 'ndecals', 'sdecals']:
        # read survey geometry
        data_path = '/fs/ess/PHS0336/data/'    
        dt = ft.read(f'{data_path}/rongpu/imaging_sys/tables/v3/nlrg_features_{region}_256.fits')
        mask_  = make_hp(256, dt['hpix'], 1.0) > 0.5
        mask   = hp.ud_grade(mask_, 1024)
    else:
        # full sky
        mask = np.ones(12*1024*1024)

    return mask*1.0, mask

    

# --- inputs
path_cl  = sys.argv[1]
path_cov = sys.argv[2]
region   = sys.argv[3]  # for window
output   = sys.argv[4]

nsteps   = 1000   # int(sys.argv[2])
ndim     = 2      # Number of parameters/dimensions
nwalkers = 10     # Number of walkers to use. It should be at least twice the number of dimensions.


ncpu = cpu_count()
print("{0} CPUs".format(ncpu))    

if not os.path.exists(os.path.dirname(output)):
    print(f'create {os.path.dirname(output)}')
    os.makedirs(os.path.dirname(output))

el_edges, cl_obs, invcov_obs = read_inputs(path_cl, path_cov)
weight, mask = read_mask(region)

z, b, dNdz = init_sample(kind='lrg')
model = SurveySpectrum()
model.add_tracer(z, b, dNdz, p=1.6)
model.make_kernels(model.el_model)
model.prep_window(weight, mask, np.arange(2048), ngauss=2048)  

lg = Posterior(model, cl_obs, invcov_obs, el_edges)
def logpost(foo):
    return lg.logpost(foo)

# scipy optimization        
#res = minimize(lg.logpost, [0., 1.0e-7], args=(cl, invcov, el), )
#print(res)
for fnl in [-10., 0., 10.]:
    print(fnl, logpost([fnl, 5.3e-7]))


# Initial positions of the walkers. TODO: add res.x
np.random.seed(SEED)
start = np.array([1.0, 1.0e-7])*np.random.randn(nwalkers, ndim) 
print(f'initial guess: {start[:2]} ... {start[-1]}')

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool)
    sampler.run_mcmc(start, nsteps, progress=True)

np.save(output, sampler.get_chain(), allow_pickle=False)
