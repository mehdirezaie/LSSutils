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
from lssutils.stats.window import WindowSHT
from lssutils.theory.cell import init_sample, Spectrum
from lssutils.extrn.mcmc import LogPosterior


class SurveySpectrumConfig(Spectrum, WindowSHT):
    el_model = np.arange(2000)
    
    def __init__(self, *arrays, **kwargs):
        Spectrum.__init__(self, *arrays, **kwargs)
        
    def add_window(self, *arrays, **kwargs):
        WindowSHT.__init__(self, *arrays, **kwargs)
        
    def __call__(self, el, fnl=0.0, b=1.0, noise=0.0):  
        cl_ = Spectrum.__call__(self, self.el_model, fnl=fnl, b=b, noise=noise)   
        clm_ = self.convolve(self.el_model, cl_)
        return clm_[el]
    

class SurveySpectrumMixm(Spectrum):
    el_model = np.arange(501)
    
    def __init__(self, *arrays, **kwargs):
        Spectrum.__init__(self, *arrays, **kwargs)
        #self.mixm = np.load('mixm_namaster.npy')[:302, :501]
        self.mixm = np.load('/fs/ess/PHS0336/data/rongpu/imaging_sys/window/window_desic_256_all.npy')
        
    def __call__(self, el, fnl=0.0, b=1.0, noise=0.0):
        cl_ = Spectrum.__call__(self, self.el_model, fnl=fnl, b=b, noise=noise)      
        clm_ = self.mixm.dot(cl_)
        return clm_[el]   



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
from argparse import ArgumentParser
ap = ArgumentParser(description='MCMC')
ap.add_argument('--path_cl', required=True)
ap.add_argument('--path_cov', required=True)
ap.add_argument('--region', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--scale', action='store_true')
ap.add_argument('--elmin', default=0, type=int)
ap.add_argument('--p', default=1.0, type=float)
ap.add_argument('--s', default=0.945, type=float)
ap.add_argument('--model')
ns = ap.parse_args()

for (key, value) in ns.__dict__.items():
    print(f'{key:15s} : {value}')                

nsteps   = 10000   # int(sys.argv[2])
ndim     = 3      # Number of parameters/dimensions
nwalkers = 50     # Number of walkers to use. It should be at least twice the number of dimensions.
assert nwalkers > 2*ndim

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))    

if not os.path.exists(os.path.dirname(ns.output)):
    print(f'create {os.path.dirname(ns.output)}')
    os.makedirs(os.path.dirname(ns.output))

el_edges, cl_obs, invcov_obs = read_inputs(ns.path_cl, ns.path_cov, ns.scale, ns.elmin)
weight, mask = read_mask(ns.region)
print('fsky', mask.mean())

if not ns.scale:
    print('using mock window')
    weight[mask] = 1.0 # if scale is not activate, then it is a mock, no fpix needed
else:
    print('using data window')

z, b, dNdz = init_sample(kind='lrg')


if ns.model == 'config':
    model = SurveySpectrumConfig()
    model.add_tracer(z, b, dNdz, p=ns.p, s=ns.s)
    model.add_kernels(model.el_model)
    model.add_window(weight, mask, np.arange(2048), ngauss=2048) 
elif ns.model == 'mixm':
    model = SurveySpectrumMixm()
    model.add_tracer(z, b, dNdz, p=ns.p, s=ns.s)
    model.add_kernels(model.el_model)    
else:
    raise ValueError(f"{ns.model} not recognized")


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

np.savez(ns.output, **{'chain':sampler.get_chain(), 
                    'log_prob':sampler.get_log_prob(), 
                    'best_fit':res.x,
                    'best_fit_logprob':res.fun,
                    'best_fit_success':res.success, 
                    '#data':cl_obs.size,
                    '#params':ndim})
