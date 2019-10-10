#
#
#import matplotlib.pyplot as plt
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
    ap = ArgumentParser(description='Neural Net regression')
    ap.add_argument('--galaxy_path',  default='/Volumes/TimeMachine/data/eboss/v6/eBOSS_QSO_clustering_NGC_v6.dat.fits')
    ap.add_argument('--random_path',  default='/Volumes/TimeMachine/data/eboss/v6/eBOSS_QSO_clustering_NGC_v6.ran.fits')
    ap.add_argument('--output_path',  default='/Volumes/TimeMachine/data/eboss/v6/results_ngc/clustering/pk_256_p8_2p2')
    ap.add_argument('--nmesh',        default=256, type=int)
    ap.add_argument('--zlim',         nargs='*',   type=float, default=[0.8, 2.2])
    ap.add_argument('--sys_tot',      action='store_true')
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
    nmesh       = ns.nmesh
    sys_tot      = ns.sys_tot 
else:
    galaxy_path = None
    random_path = None
    output_path = None
    zlim        = None
    nmesh       = None
    sys_tot     = None


galaxy_path = comm.bcast(galaxy_path, root=0)
random_path = comm.bcast(random_path, root=0)
output_path = comm.bcast(output_path, root=0)
zlim = comm.bcast(zlim, root=0) 
nmesh = comm.bcast(nmesh, root=0)
sys_tot = comm.bcast(sys_tot, root=0)  

# 
data    = nb.FITSCatalog(galaxy_path)
randoms = nb.FITSCatalog(random_path)

if rank == 0:    
    print('data    columns = ',    data.columns, data.size)
    print('randoms columns = ', randoms.columns, randoms.size)


ZMIN = zlim[0]
ZMAX = zlim[1]


# slice the data and randoms
valid = (data['Z'] > ZMIN) & (data['Z'] < ZMAX)
data  = data[valid]
valid    = (randoms['Z'] > ZMIN) & (randoms['Z'] < ZMAX)
randoms  = randoms[valid]



# the fiducial BOSS DR12 cosmology
cosmo = Cosmology(h=0.676).match(Omega0_m=0.31)

# add Cartesian position column
data['Position']    = SkyToCartesian(data['RA'],    data['DEC'],    data['Z'],    cosmo=cosmo)
randoms['Position'] = SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

# apply the Completeness weights to both data and randoms
if sys_tot:
    if rank ==0:print('including sys_tot')
    data['WEIGHT']      = data['WEIGHT_SYSTOT']    * data['WEIGHT_NOZ']    * data['WEIGHT_CP']
else:
    if rank ==0:print('excluding sys_tot')    
    data['WEIGHT']      = data['WEIGHT_NOZ']    * data['WEIGHT_CP'] # data['WEIGHT_SYSTOT'] 
    
randoms['WEIGHT']   = randoms['WEIGHT_SYSTOT'] * randoms['WEIGHT_NOZ'] * randoms['WEIGHT_CP']

# combine the data and randoms into a single catalog
fkp  = nb.FKPCatalog(data, randoms, nbar='NZ')
mesh = fkp.to_mesh(Nmesh=nmesh, fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc')

# compute 
r = nb.ConvolvedFFTPower(mesh, poles=[0, 2, 4], dk=0.001, kmin=0.0)

# save
r.save(output_path)
