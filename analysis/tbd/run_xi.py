

import numpy as np
from nbodykit.transform import SkyToCartesian
from nbodykit.cosmology import Cosmology
import nbodykit.lab as nb
from nbodykit import setup_logging, style
setup_logging() # turn on logging to screen


#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()   

from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size

if rank ==0:
    import os
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Correlation Function (NBODYKIT/Corrfunc)')
    ap.add_argument('-g', '--galaxy_path',  default='/Volumes/TimeMachine/data/eboss/v6/eBOSS_QSO_clustering_NGC_v6.dat.fits')
    ap.add_argument('-r', '--random_path',  default='/Volumes/TimeMachine/data/eboss/v6/eBOSS_QSO_clustering_NGC_v6.ran.fits')
    ap.add_argument('-o', '--output_path',  default='/Volumes/TimeMachine/data/eboss/v6/results_ngc/clustering/xi_256_p8_2p2.json')
    ap.add_argument('--zlim',         nargs='*',   type=float, default=[0.8, 2.2])
    ap.add_argument('--use_systot',      action='store_true')
    ns = ap.parse_args()

    outpath1 = ns.output_path.split('/')
    outpath1 = '/'.join(outpath1[:-1])
    if not os.path.isdir(outpath1):
        os.makedirs(outpath1)


    args = ns.__dict__
    for (a,b) in zip(args.keys(), args.values()):
        print('{:6s}{:15s} : {}\n'.format('', a, b)) 

    galaxy_path = ns.galaxy_path
    random_path = ns.random_path
    output_path = ns.output_path
    zlim        = ns.zlim
    use_systot      = ns.use_systot 
else:
    galaxy_path = None
    random_path = None
    output_path = None
    zlim        = None
    use_systot     = None


galaxy_path = comm.bcast(galaxy_path, root=0)
random_path = comm.bcast(random_path, root=0)
output_path = comm.bcast(output_path, root=0)
zlim = comm.bcast(zlim, root=0) 
use_systot = comm.bcast(use_systot, root=0)  

# 
data    = nb.FITSCatalog(galaxy_path)
randoms = nb.FITSCatalog(random_path)

if rank == 0:    
    print('data    columns = ',    data.columns, data.size)
    print('randoms columns = ', randoms.columns, randoms.size)


ZMIN = zlim[0]
ZMAX = zlim[1]

mode = '2d'      # r, mu
dr = 10.0       # Mpc/h
nmu = 120
rmin = 1.e-15
rmax = 200.0
edges = np.arange(rmin, rmax+2*dr, dr)

# slice the data and randoms
compmin=0.5
valid = (data['Z'] >= ZMIN) & (data['Z'] <= ZMAX)
if 'IMATCH' in data.columns:
    valid &= (data['IMATCH']==1) | (data['IMATCH']==2)
if 'COMP_BOSS' in data.columns:
    valid &= data['COMP_BOSS'] > compmin
if 'sector_SSR' in data.columns:
    valid &= data['sector_SSR'] > compmin
data  = data[valid]

valid  = (randoms['Z'] >= ZMIN) & (randoms['Z'] <= ZMAX)
if 'COMP_BOSS' in randoms.columns:
    valid &= randoms['COMP_BOSS']  > compmin
if 'sector_SSR' in randoms.columns:
    valid &= randoms['sector_SSR'] > compmin
randoms  = randoms[valid]

# the fiducial BOSS DR12 cosmology
cosmo = Cosmology(h=0.676).match(Omega0_m=0.31)

# apply the Completeness weights to both data and randoms
if use_systot:
    if rank ==0:print('including sys_tot')
    data['WEIGHT'] = data['WEIGHT_SYSTOT'] * data['WEIGHT_NOZ'] * data['WEIGHT_CP'] * data['WEIGHT_FKP']  
else:
    if rank ==0:print('excluding sys_tot')    
    data['WEIGHT'] = data['WEIGHT_NOZ']    * data['WEIGHT_CP'] * data['WEIGHT_FKP']
    
randoms['WEIGHT']   = randoms['WEIGHT_SYSTOT'] * randoms['WEIGHT_NOZ'] * randoms['WEIGHT_CP'] * randoms['WEIGHT_FKP']



results = nb.SurveyData2PCF(mode, data, randoms, edges, cosmo=cosmo,
                            Nmu=nmu, ra='RA', dec='DEC', redshift='Z',
                            weight='WEIGHT', show_progress=True)

results.save(output_path)     # uncomment to save as json
comm.Barrier()

#   write the result to ascii
if rank==0:
    
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
    print('writing ... %s'%output_path.replace('.json', '.txt'))
    with open(output_path.replace('.json', '.txt'), 'w') as of:

        # write the attributes
        for k in results.attrs:
            if k=='edges':
                continue
            of.write(f'#{k:20s} : {results.attrs[k]}\n')
        
        of.write('# r_mid - xi_0 - xi_1 - xi_2 - xi_3 - xi_4\n')
        for i in range(corr.shape[0]):
            of.write('{0} {1} {2} {3} {4} {5}\n'.format(rmid[i], poles['corr_0'][i],
                                                        poles['corr_1'][i], poles['corr_2'][i], 
                                                        poles['corr_3'][i], poles['corr_4'][i]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    ls = 2*['-', '--', '-.', ':']
    r  = rmid
    for i,ell in enumerate([0, 2, 4]):
        plt.plot(r, r*r*poles['corr_%d'%ell], 
                label=r'$\ell$=%d'%ell, linestyle=ls[i])
    plt.ylabel(r'$r^{2}\xi_{\ell}$')
    plt.xlabel('r [Mpc/h]')
    plt.legend()
    plt.savefig(output_path.replace('.json', '.pdf'), bbox_inches='tight')
    print('plotting done')
