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
from lssutils.extrn.mcmc import LogPosterior

SEED = 85

def load_cl(path_cl):
    d = np.load(path_cl, allow_pickle=True)
    if hasattr(d, 'files'):
        return d
    else:
        from lssutils.utils import histogram_cell, ell_edges
        cl = d.item()
        lb, clb = histogram_cell(cl['cl_gg']['l'], cl['cl_gg']['cl'], bins=ell_edges)
        return {'el_edges':ell_edges, 'el_bin':lb, 'cl':np.log10(clb)}

def read_inputs(path_cl, path_cov, scale=True, elmin=0):
    
    dcl_obs = load_cl(path_cl)
    dclcov_obs = np.load(path_cov)
    assert np.array_equal(dcl_obs['el_edges'], dclcov_obs['el_edges'])
    
    el_edges = dcl_obs['el_edges'][elmin:]
    cl_obs = dcl_obs['cl'][elmin:]
    clcov = dclcov_obs['clcov'][elmin:,][:, elmin:]
    if scale:
        clcov *= 1000.
    
    print(el_edges[:10], len(el_edges))
    print(cl_obs[:10], len(cl_obs))
    print(clcov[:10, :][:, :10], clcov.shape)

    invcov_obs = np.linalg.inv(clcov)
    return el_edges, cl_obs, invcov_obs

def read_mask(region):

    import fitsio as ft
    import healpy as hp
    from lssutils.utils import make_hp

    # read survey geometry
    data_path = f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_{region}_256.fits'
    if os.path.exists(data_path):
        dt = ft.read(data_path)
        weight_ = make_hp(256, dt['hpix'], dt['fracgood'])
        weight  = hp.ud_grade(weight_, 1024)        
    else:
        # full sky
        weight = np.ones(12*1024*1024, 'f8')
    print('fsky', weight.mean())
    return weight, weight > 0

    

# --- inputs
path_cl  = sys.argv[1]
path_cov = sys.argv[2]
region   = sys.argv[3]  # for window
output   = sys.argv[4]
scale    = float(sys.argv[5]) > 0
elmin    = int(sys.argv[6])

nsteps   = 10000   # int(sys.argv[2])
ndim     = 3      # Number of parameters/dimensions
nwalkers = 50     # Number of walkers to use. It should be at least twice the number of dimensions.
assert nwalkers > 2*ndim

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))    
print("scale it", scale)
print("elmin bin id", elmin) 
if not os.path.exists(os.path.dirname(output)):
    print(f'create {os.path.dirname(output)}')
    os.makedirs(os.path.dirname(output))

el_edges, cl_obs, invcov_obs = read_inputs(path_cl, path_cov, scale, elmin)
weight, mask = read_mask(region)
print('fsky', mask.mean())

if not scale:
    print('using mock window')
    weight[mask] = 1.0 # if scale is not activate, then it is a mock, no fpix needed
else:
    print('using data window')

z, b, dNdz = init_sample(kind='lrg')
model = SurveySpectrum()
model.add_tracer(z, b, dNdz, p=1.0)
model.add_kernels(model.el_model)
model.add_window(weight, mask, np.arange(2048), ngauss=2048)  

lg = LogPosterior(model, cl_obs, invcov_obs, el_edges)
def logpost(params):
    return lg.logpost(params)
def neglogpost(params):
    return -lg.logpost(params)


# scipy optimization        
np.random.seed(SEED)
res = minimize(neglogpost, [1.0, 1.0, 1.0e-7], method='Powell')

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
                    '#data':cl_obs.size,
                    '#params':ndim})
