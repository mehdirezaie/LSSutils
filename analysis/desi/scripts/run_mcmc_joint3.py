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
from argparse import ArgumentParser
ap = ArgumentParser(description='MCMC')
ap.add_argument('--path_cl', nargs=3, required=True)
ap.add_argument('--path_cov', nargs=3, required=True)
ap.add_argument('--region', nargs=3, required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--scale', action='store_true')
ap.add_argument('--p', default=1.0, type=float)
ap.add_argument('--s', nargs=3, default=[0.945, 0.945, 0.945], type=float)
ns = ap.parse_args()

for (key, value) in ns.__dict__.items():
    print(f'{key:15s} : {value}')                


nsteps   = 10000   # int(sys.argv[2])
ndim     = 7      # Number of parameters/dimensions
nwalkers = 50     # Number of walkers to use. It should be at least twice the number of dimensions.
assert nwalkers > 2*ndim

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))    

if not os.path.exists(os.path.dirname(ns.output)):
    print(f'create {os.path.dirname(ns.output)}')
    os.makedirs(os.path.dirname(ns.output))

z, b, dNdz = init_sample(kind='lrg')

# region 1
el_edges, cl_obs, invcov_obs = read_inputs(ns.path_cl[0], ns.path_cov[0], ns.scale)
weight, mask = read_mask(ns.region[0])

model = SurveySpectrum()
model.add_tracer(z, b, dNdz, p=ns.p, s=ns.s[0])
model.add_kernels(model.el_model)
model.add_window(weight, mask, np.arange(2048), ngauss=2048)  

lg = LogPosterior(model, cl_obs, invcov_obs, el_edges)

# region 2
el_edges2, cl_obs2, invcov_obs2 = read_inputs(ns.path_cl[1], ns.path_cov[1], ns.scale)
weight, mask = read_mask(ns.region[1])

model2 = SurveySpectrum()
model2.add_tracer(z, b, dNdz, p=ns.p, s=ns.s[1])
model2.add_kernels(model2.el_model)
model2.add_window(weight, mask, np.arange(2048), ngauss=2048)  

lg2 = LogPosterior(model2, cl_obs2, invcov_obs2, el_edges2)

# region 3
el_edges3, cl_obs3, invcov_obs3 = read_inputs(ns.path_cl[2], ns.path_cov[2], ns.scale)
weight, mask = read_mask(ns.region[2])

model3 = SurveySpectrum()
model3.add_tracer(z, b, dNdz, p=ns.p, s=ns.s[2])
model3.add_kernels(model3.el_model)
model3.add_window(weight, mask, np.arange(2048), ngauss=2048)  

lg3 = LogPosterior(model3, cl_obs3, invcov_obs3, el_edges3)

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

np.savez(ns.output, **{'chain':sampler.get_chain(), 
                    'log_prob':sampler.get_log_prob(), 
                    'best_fit':res.x,
                    'best_fit_logprob':res.fun,
                    'best_fit_success':res.success, 
                    '#data':cl_obs.size+cl_obs2.size+cl_obs3.size,
                    '#params':ndim})
