

from LSSutils import CurrentMPIComm
from nbodykit.transform import SkyToCartesian
from nbodykit.cosmology import Cosmology
import nbodykit.lab as nb



__all__ = ['power']

@CurrentMPIComm.enable
def power(data, randoms, zlim=[0.8, 2.2], poles=[0, 2, 4],
          dk=0.001, sys_tot=True, nmesh=512, comm=None):
    
    data = nb.ArrayCatalog(data)
    randoms = nb.ArrayCatalog(randoms)
    
    if comm.rank == 0:    
        print('data    columns = ',    data.columns)#data.size)
        print('randoms columns = ', randoms.columns)#, randoms.size)


    ZMIN = zlim[0]
    ZMAX = zlim[1]


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

    # add Cartesian position column
    data['Position']    = SkyToCartesian(data['RA'],    data['DEC'],    data['Z'],    cosmo=cosmo)
    randoms['Position'] = SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

    # apply the Completeness weights to both data and randoms
    if sys_tot:
        if comm.rank ==0:print('including sys_tot')
        data['WEIGHT']      = data['WEIGHT_SYSTOT']    * data['WEIGHT_NOZ']    * data['WEIGHT_CP']
    else:
        if comm.rank ==0:print('excluding sys_tot')    
        data['WEIGHT']      = data['WEIGHT_NOZ']    * data['WEIGHT_CP'] # data['WEIGHT_SYSTOT'] 

    randoms['WEIGHT']   = randoms['WEIGHT_SYSTOT'] * randoms['WEIGHT_NOZ'] * randoms['WEIGHT_CP']

    # combine the data and randoms into a single catalog
    fkp  = nb.FKPCatalog(data, randoms, nbar='NZ')
    mesh = fkp.to_mesh(Nmesh=nmesh, fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc')

    # compute 
    r = nb.ConvolvedFFTPower(mesh, poles=poles, dk=dk, kmin=0.0)
    
    return r

# # save
# r.save(output_path)

# # --- save in txt format
# comm.Barrier()

# if rank == 0:
#     #
#     # write P0-shotnoise P2 P4 to file
#     with open(output_path.replace('.json', '.txt'), 'w') as output:

#         # write the attributes
#         for k in r.attrs:
#             output.write(f'#{k:20s} : {r.attrs[k]}\n')

#         nbins = len(r.poles.coords['k'])
#         output.write('# kmid, kavg, P0, P2, P4, Nmodes\n')
#         for i in range(nbins):
#            output.write('{} {} {} {} {} {}\n'.format(r.poles.coords['k'][i], 
#                                                      r.poles['k'][i], 
#                                                      r.poles['power_0'][i].real, 
#                                                      r.poles['power_2'][i].real, 
#                                                      r.poles['power_4'][i].real, 
#                                                      r.poles['modes'][i]))

#     #
#     if ns.plot:
#         # plot k vs kPell
#         import matplotlib
#         matplotlib.use('Agg')
#         import matplotlib.pyplot as plt

#         k = r.poles['k']
#         plt.figure()
#         plt.title(f'{ns.zlim}')
#         for ell in [0, 2, 4]:
#             pk = r.poles['power_%d'%ell].real
#             if ell == 0:pk -= r.poles.attrs['shotnoise']
#             plt.plot(k, k*pk, label=r'$\ell$=%d'%ell)
#         plt.ylabel(r'kP$_{\ell}$(k)')
#         plt.xlabel('k [h/Mpc]')
#         plt.legend()
#         plt.savefig(output_path.replace('.json', '.pdf'), bbox_inches='tight')
#     print('plot and txt done!')

