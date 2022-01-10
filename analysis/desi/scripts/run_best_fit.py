"""
    Best Fit Optimizer for Angular Power Spectra of Mocks 

"""
import sys
import os
import emcee
import numpy as np

from scipy.optimize import minimize
from time import time
from lssutils.utils import split_NtoM, histogram_cell
from lssutils.theory.cell import dNdz_model, init_sample, SurveySpectrum
from lssutils.extrn.mcmc import Posterior

from mpi4py import MPI
 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# read covariance and cls with root
if rank==0:
    region = sys.argv[1]  # fullsky, bmzls, ndecals, sdecals ##for window
    method = 'noweight'
    output = f'/fs/ess/PHS0336/data/lognormal/v0/mcmc/bestfit_{region}_{method}_fine.npz'

    if not os.path.exists(os.path.dirname(output)):
        print(f'create {os.path.dirname(output)}')
        os.makedirs(os.path.dirname(output))
    print('will output ', output)

    if region=='fullsky':
        path_cl  = '/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_fullsky.npy'
        path_cov = '/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_fullsky_cov_fine.npz'
        cl_mocks = np.load(path_cl)
        
        weight = np.ones(12*1024*1024, 'f8') # full sky
    else:
        import fitsio as ft
        import healpy as hp
        from glob import glob
        from lssutils.utils import hpix2radec, radec2hpix

        path_cl  = f'/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_*_lrg_{region}_256_{method}.npy'
        path_cov = f'/fs/ess/PHS0336/data/lognormal/v0/clustering/clmock_{region}_{method}_cov_fine.npz'    
        
        cl_mocks = []
        for fl in glob(path_cl):
            cl_mocks.append(np.load(fl, allow_pickle=True).item()['cl_gg']['cl'])
        cl_mocks = np.array(cl_mocks)

        dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v3/nlrg_features_{region}_256.fits')
        weight_ = np.zeros(12*256*256, 'f8')
        weight_[dt['hpix']] = 1.0
        weight = hp.ud_grade(weight_, 1024)



    print('cl mocks shape:', cl_mocks.shape)
    print('sky coverage:', weight.mean())
    cl_cov_ = np.load(path_cov)
    el_edges = cl_cov_['el_edges']
    icov = np.linalg.inv(cl_cov_['clcov'])
    
    zbdndz = init_sample(kind='lrg')
else:
    zbdndz = None
    cl_mocks = None
    el_edges = None
    icov = None
    weight = None

zbdndz = comm.bcast(zbdndz, root=0)
cl_mocks = comm.bcast(cl_mocks, root=0)
el_edges = comm.bcast(el_edges, root=0)
icov = comm.bcast(icov, root=0)
weight = comm.bcast(weight, root=0)
mask = weight > 0

model = SurveySpectrum()
model.add_tracer(*zbdndz, p=1.6)
model.add_kernels(model.el_model)
model.add_window(weight, mask, np.arange(2048), ngauss=2048)

nmocks, elmax = cl_mocks.shape
el = np.arange(elmax)
start, end = split_NtoM(nmocks, size, rank)

params = []
neglog = []
success = []

for i in range(start, end+1):
    cl_i = cl_mocks[i, :]
    cl_obs = histogram_cell(el, cl_i, bins=el_edges)[1]

    lg = Posterior(model, cl_obs, icov, el_edges)
    def neglogpost(foo):
        return -lg.logpost(foo)

    res = minimize(neglogpost, [1.0, 1.0, 1.0e-7], method='Powell')

    params.append(res.x)
    neglog.append(res.fun)
    success.append(res.success)


comm.Barrier()

params = comm.gather(params, root=0)
neglog = comm.gather(neglog, root=0)
success = comm.gather(success, root=0)

if rank==0:
    np.savez(output, **{'params':np.concatenate(np.array(params)), 
                        'success':np.concatenate(success),
                        'neglog':np.concatenate(neglog)})
