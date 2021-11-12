'''

   Code to compute the 2pc for a Simulation Box



    Updates:
    Sep 3, 2019: modify ``/home/mehdi/miniconda3/envs/\
                        py3p6/lib/python3.7/site-packages/\
                        nbodykit/algorithms/pair_counters/base.py''
                        to include self pairs

'''
import sys
import numpy as np
import nbodykit.lab as nb               # Import Nbodykit
from nbodykit import setup_logging
from time import time

setup_logging("info")                   # Info

from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size

#  I/O 
if rank == 0:
    input_name  = sys.argv[1]
    ouname1     = sys.argv[2]
    ouname2     = sys.argv[3]
    action      = sys.argv[4]
    box     = float(sys.argv[5])
else:
    input_name  = None
    ouname1     = None
    ouname2     = None
    action      = None
    box     = None





# bcast
input_name  = comm.bcast(input_name,  root=0)
ouname1     = comm.bcast(ouname1,     root=0) 
ouname2     = comm.bcast(ouname2,     root=0) 
action      = comm.bcast(action,      root=0)
box         = comm.bcast(box,     root=0)

# Input parameters
mode       = '2d'      # r, mu
dr         = 1.0       # Mpc/h
nmu        = 120
rmin       = 0.0
rmax       = 200.0
#box        = 1000      # Mpc/h 
edges      = np.arange(rmin, rmax+2*dr, dr)

#MOCKNAME   = 'UNIT'
#LASTNAME   = 'REZAIE'
#BINNING    = 'lin'
#ESTIMATOR1 = 'xi2D'
#ESTIMATOR2 = 'xil'
#version    = 1
#ouname1    = f'{ESTIMATOR1}_{BINNING}_{LASTNAME}_{MOCKNAME}_{version}.txt'
#ouname2    = f'{ESTIMATOR2}_{BINNING}_{LASTNAME}_{MOCKNAME}_{version}.txt'





# read data
catalog = nb.FITSCatalog(input_name)
for col in ['x', 'y', 'z', 'z_rsd']:
    if not col in catalog.columns:raise RuntimeError('%s not available'%col)

# redshift or real space
if action == 'ddrmu':
    catalog['Position'] = np.column_stack([catalog['x'], catalog['y'], 
                                          catalog['z']])
elif action == 'ddsmu':
    catalog['Position'] = np.column_stack([catalog['x'], catalog['y'], 
                                           catalog['z_rsd']])

# run Corrfunc
if rank==0:t0=time( )
results = nb.SimulationBox2PCF(mode, catalog, edges, Nmu=nmu,
                               periodic=True, BoxSize=box, los='z', 
                               weight='Weight', position='Position', 
                               show_progress=True, xbin_refine_factor=2, 
                               ybin_refine_factor=2, zbin_refine_factor=2)  # for PMILL
results.save(ouname1.replace('.txt', '.json'))     # uncomment to save as json
comm.Barrier()

#   write the result to ascii
if rank==0:
    print('took {} sec'.format(time()-t0))
    dd   = results.D1D2['npairs']
    r    = results.corr.edges['r']
    mu   = results.corr.edges['mu']
    corr = results.corr.data['corr']
    poles = results.corr.to_poles([0, 2, 4])

    # s, mu, corr, DD
    print('writing ... %s'%ouname1)
    of = open(ouname1, 'w')
    of.write('# r_min - mu_min - xi - DD\n')
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            of.write('{0} {1} {2} {3}\n'.format(r[i], mu[j], corr[i, j], dd[i, j]))
    of.close()

    # s, xi_0, xi_2, xi_4
    print('writing ... %s'%ouname2)
    of = open(ouname2, 'w')
    of.write('# r_min - xi_0 - xi_2 - xi_4\n')
    for i in range(corr.shape[0]):
        of.write('{0} {1} {2} {3}\n'.format(r[i], poles['corr_0'][i], poles['corr_2'][i], poles['corr_4'][i]))
    of.close()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    ls = ['-', '--', '-.']
    r  = r[:-1]    # r_min 
    for i,ell in enumerate([0, 2, 4]):
        plt.plot(r, r*r*poles['corr_%d'%ell], 
                label=r'$\ell$=%d'%ell, linestyle=ls[i])
    plt.ylabel(r'$r^{2}\xi_{\ell}$')
    plt.xlabel('r [Mpc/h]')
    plt.legend()
    plt.savefig(ouname2.replace('.txt', '.pdf'), bbox_inches='tight')


