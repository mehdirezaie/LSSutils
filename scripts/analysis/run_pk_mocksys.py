'''

    Use Nbodykit to compute P0


    - Jan 16: add nmodes to output
    - Nov 28: The functionality for reading multiple randoms does not work
'''
import sys
import nbodykit.lab as nb
import numpy as np

from nbodykit.transform import SkyToCartesian
from scipy.interpolate import InterpolatedUnivariateSpline
from nbodykit.cosmology import Planck15 as cosmo
from nbodykit import setup_logging

setup_logging("info")

from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size

#  I/O 
if rank == 0:
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Power Spectrum')
    ap.add_argument('--data',    default='/B/Shared/Shadab/FA_LSS/FA_EZmock_desi_ELG_v0_15.fits')
    ap.add_argument('--randoms', nargs='*', type=str, default='/B/Shared/Shadab/FA_LSS/FA_EZmock_desi_ELG_v0_rand_0*.fits')
    #ap.add_argument('--randoms', default='/B/Shared/Shadab/FA_LSS/FA_EZmock_desi_ELG_v0_rand_01.fits')
    ap.add_argument('--mask',    default='None')
    ap.add_argument('--output',  default='/home/mehdi/data/mocksys/pk_v0_15.txt')
    ap.add_argument('--nmesh',   default=512, type=int) # v0.1 256
    ap.add_argument('--zlim',    nargs='*', type=float, default=[0.7, 1.5]) # v0.1 [0.7, 0.9]
    ap.add_argument('--poles',   nargs='*', type=int, default=[0, 2, 4])
    ap.add_argument('--real',    action='store_true')
    ns = ap.parse_args()
else:
    ns = None
    
ns  = comm.bcast(ns, root=0)

if rank ==0:
    args = ns.__dict__
    for (a,b) in zip(args.keys(), args.values()):
        print('{:6s}{:15s} : {}'.format('', a, b))
        

        
ZMIN, ZMAX = ns.zlim

if ns.mask != 'None':
    mask    = nb.FITSCatalog(ns.mask)
else:
    mask    = None


        

data    = nb.FITSCatalog(ns.data)
randoms = nb.FITSCatalog(ns.randoms)


if ns.real:
    zcol_name = 'Z_COSMO'
else:
    zcol_name = 'Z_RSD'
    data[zcol_name]    = data['Z_COSMO'] + data['DZ_RSD']
    randoms[zcol_name] = randoms['Z_COSMO'] + randoms['DZ_RSD']
    

if rank == 0:
    print('Only Rank %d'%rank)
    print('Data : ', data.columns, data.csize, data.size)
    print('Randoms : ', randoms.columns, randoms.csize, randoms.size)    


# slice the data and randoms
if mask is None:
    valid = (data[zcol_name] > ZMIN)&(data[zcol_name] < ZMAX)
else:
    valid = (data[zcol_name] > ZMIN)&(data[zcol_name] < ZMAX)\
            & (mask['bool_index'])
    
data = data[valid]


valid   = (randoms[zcol_name] > ZMIN)&(randoms[zcol_name] < ZMAX)
randoms = randoms[valid]

#print(data.size, randoms.size)
#sys.exit()
data['Position']    = SkyToCartesian(data['RA'],    
                                     data['DEC'],    
                                     data[zcol_name],    
                                     cosmo=cosmo)
randoms['Position'] = SkyToCartesian(randoms['RA'], 
                                     randoms['DEC'], 
                                     randoms[zcol_name],
                                     cosmo=cosmo)

# Get N(z)
# the sky fraction, used to compute volume in n(z)
FSKY  = 0.37 # a made-up value
zhist = nb.RedshiftHistogram(randoms, FSKY, cosmo, redshift=zcol_name)
alpha = 1.0 * data.csize / randoms.csize
nofz  = InterpolatedUnivariateSpline(zhist.bin_centers, alpha*zhist.nbar)


# initialize the FKP source
# add the n(z) columns to the FKPCatalog
randoms['NZ'] = nofz(randoms[zcol_name])
data['NZ']    = nofz(data[zcol_name])
fkp = nb.FKPCatalog(data, randoms, nbar='NZ')


Pzero  = 6000
fkp['data/FKPWeight']    = 1.0 / (1 + fkp['data/NZ'] * Pzero)
fkp['randoms/FKPWeight'] = 1.0 / (1 + fkp['randoms/NZ'] * Pzero)


mesh = fkp.to_mesh(Nmesh=ns.nmesh, 
                   fkp_weight='FKPWeight', 
                   comp_weight='Weight', 
                   window='tsc')
r    = nb.ConvolvedFFTPower(mesh, poles=ns.poles, dk=0.001, kmin=0.0)


comm.Barrier()
if rank == 0:
    #
    # write P0-shotnoise P2 P4 to file
    output = open(ns.output, 'w')

    # write the attributes
    for k in r.attrs:
        output.write(f'#{k:20s} : {r.attrs[k]}\n')
    
    output.write('# k_avg P0 P2 P4 Nmodes\n')
    for i in range(r.poles['k'].size):
        output.write('{} {} {} {} {}\n'.format(r.poles['k'][i], 
                                               r.poles['power_0'][i].real-r.poles.attrs['shotnoise'], 
                                               r.poles['power_2'][i].real, 
                                               r.poles['power_4'][i].real,
                                               r.poles['modes'][i]))   

    # plot k vs kPell
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    k = r.poles['k']
    plt.figure()
    plt.title(zcol_name)
    for ell in [0, 2, 4]:
        pk = r.poles['power_%d'%ell].real
        if ell == 0:pk -= r.poles.attrs['shotnoise']
        plt.plot(k, k*pk, label=r'$\ell$=%d'%ell)
    plt.ylabel(r'kP$_{\ell}$(k)')
    plt.xlabel('k [h/Mpc]')
    plt.savefig(ns.output.replace('.txt', '.pdf'))
    print('plot done!')
