'''

   Code to compute the 2pc for a Survey Catalog



    Updates:
    Sep 3, 2019: modify ``/home/mehdi/miniconda3/envs/\
                        py3p6/lib/python3.7/site-packages/\
                        nbodykit/algorithms/pair_counters/base.py''
                        to include self pairs

'''
import sys
import numpy as np
import nbodykit.lab as nb               # Import Nbodykit
from nbodykit.cosmology import Planck15 as cosmo
from nbodykit import setup_logging
from time import time

setup_logging("info")                   # Info

from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size

#  I/O 
if rank == 0:
    space = sys.argv[1]
else:
    space = None

   
space      = comm.bcast(space,      root=0)

if space not in ['real', 'red']:
    raise ValueError('space not valid')
 

# Input parameters
path       = '/home/mehdi/data/step2/'
mode       = '2d'      # r, mu
dr         = 1.0       # Mpc/h
nmu        = 120
rmin       = 1.e-15
rmax       = 200.0
#box        = 1000      # Mpc/h 
edges      = np.arange(rmin, rmax+2*dr, dr)

MOCKNAME   = 'UNIT'
LASTNAME   = 'REZAIE'
BINNING    = 'lin'

ESTIMATOR2 = 'xil'
version    = 2

ouname2    = path + f'{ESTIMATOR2}_{BINNING}_{LASTNAME}_{MOCKNAME}_{version}_{space}.txt'
input_name2= path + 'UNIT_lightcone_ELG_rand_NGC_Z_rsd.dat'
input_name1= path + 'UNIT_lightcone_ELG_NGC_Z_rsd.dat'

    
# read data
data1    = nb.CSVCatalog(input_name1, ['RA','DEC', 'Z', 'Z_RSD'])
randoms1 = nb.CSVCatalog(input_name2, ['RA','DEC', 'Z', 'Z_RSD'])

# test
#data1    = data1[:10000]
#randoms1 = randoms1[:10000]


# redshift or real space
if space == 'real':
    redname ='Z'
elif space == 'red':
    redname = 'Z_RSD'    
    
# run Corrfunc
if rank==0:t0=time( )
    
    
results = nb.SurveyData2PCF(mode, data1, randoms1, edges, cosmo=cosmo,
                            Nmu=nmu, ra='RA', dec='DEC', redshift=redname,
                            weight='Weight', show_progress=True)

results.save(ouname2.replace('.txt', '.json'))     # uncomment to save as json
comm.Barrier()

#   write the result to ascii
if rank==0:
    print('took {} sec'.format(time()-t0))
    
    r    = results.corr.edges['r']
    rmid = 0.5*(r[:-1]+r[1:])
    mu   = results.corr.edges['mu']
    corr = results.corr.data['corr']
    poles = results.corr.to_poles([0, 1, 2, 3, 4])

    # s, mu, corr
    #print('writing ... %s'%ouname1)
    #of = open(ouname1, 'w')
    #of.write('# r_min - mu_min - xi - DD\n')
    #for i in range(corr.shape[0]):
    #    for j in range(corr.shape[1]):
    #        of.write('{0} {1} {2} {3}\n'.format(r[i], mu[j], corr[i, j], dd[i, j]))
    #of.close()

    # s, xi_0, xi_2, xi_4
    print('writing ... %s'%ouname2)
    of = open(ouname2, 'w')
    of.write('# r_mid - xi_0 - xi_1 - xi_2 - xi_3 - xi_4\n')
    for i in range(corr.shape[0]):
        of.write('{0} {1} {2} {3} {4} {5}\n'.format(rmid[i], poles['corr_0'][i],
                                                    poles['corr_1'][i], poles['corr_2'][i], 
                                                    poles['corr_3'][i], poles['corr_4'][i]))
    of.close()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    ls = 2*['-', '--', '-.', ':']
    r  = rmid
    for i,ell in enumerate([0, 1, 2, 3, 4]):
        plt.plot(r, r*r*poles['corr_%d'%ell], 
                label=r'$\ell$=%d'%ell, linestyle=ls[i])
    plt.ylabel(r'$r^{2}\xi_{\ell}$')
    plt.xlabel('r [Mpc/h]')
    plt.legend()
    plt.savefig(ouname2.replace('.txt', '.pdf'), bbox_inches='tight')
