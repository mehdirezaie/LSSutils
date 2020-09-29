
import os

import nbodykit.lab as nb

from nbodykit.source.catalog import FITSCatalog
from nbodykit.transform import SkyToCartesian
from nbodykit.cosmology import Cosmology
from nbodykit import CurrentMPIComm







__all__ = ['run_ConvolvedFFTPower']

@CurrentMPIComm.enable
def run_ConvolvedFFTPower(galaxy_path, 
                          random_path,
                          output_path,
                          use_systot=True, 
                          zmin=None, 
                          zmax=None, 
                          compmin=0.5,
                          cosmo=None,
                          boxsize=None,
                          nmesh=512,
                          dk=0.002,
			  kmax=None,
                          comm=None,
                          return_pk=False):
              
        
    if cosmo is None:
        # the fiducial BOSS DR12 cosmology 
        # see Alam et al. https://arxiv.org/abs/1607.03155        
        cosmo = Cosmology(h=0.676).match(Omega0_m=0.31)

    data = FITSCat(galaxy_path)
    random = FITSCat(random_path)
    
    # select based on redshift and comp_BOSS
    kwargs = dict(compmin=compmin, zmin=zmin, zmax=zmax)
    data = data.trim_redshift_range(**kwargs)
    random = random.trim_redshift_range(**kwargs)
    
    # sky to xyz
    data.sky_to_xyz(cosmo)
    random.sky_to_xyz(cosmo)
    
    # prepare weights
    data.apply_weight(use_systot=use_systot)
    random.apply_weight(use_systot=True)  # always weight randoms by systot
    
    # combine the data and randoms into a single catalog
    mesh_kwargs = {'interlaced': True,'window': 'tsc'}
    fkp = nb.FKPCatalog(data, random, nbar='NZ', BoxSize=boxsize)
    mesh = fkp.to_mesh(Nmesh=nmesh, fkp_weight='WEIGHT_FKP', **mesh_kwargs)    
    
    r = nb.ConvolvedFFTPower(mesh, poles=[0, 2], dk=dk, kmin=0.0, kmax=kmax)
    
    comm.Barrier()    
    if comm.rank == 0:        
        for parameter in ['sigma8',  'Omega0_m', 'h', 'n_s',
                          'Omega0_b', 'Omega0_lambda']:
            r.attrs[parameter] = getattr(cosmo, parameter)
    
    if comm.rank == 0:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  
            
    r.save(output_path) 
    
    if return_pk:
        if comm.rank == 0:
            return r
    
    # --- save in txt format
    # comm.Barrier()
    # if comm.rank == 0:
    #     #
    #     # write P0-shotnoise P2 P4 to file
    #     with open(result_path.replace('.json', '.txt'), 'w') as output:

    #         # write the attributes
    #         for k in r.attrs:
    #             output.write(f'#{k:20s} : {r.attrs[k]}\n')

    #         nbins = len(r.poles.coords['k'])
    #         output.write('# kmid, kavg, P0, P2, P4, Nmodes\n')
    #         for i in range(nbins):
    #             output.write('{} {} {} {} {} {}\n'.format(r.poles.coords['k'][i], 
    #                                                      r.poles['k'][i], 
    #                                                      r.poles['power_0'][i].real, 
    #                                                      r.poles['power_2'][i].real, 
    #                                                      r.poles['power_4'][i].real, 
    #                                                      r.poles['modes'][i]))



class FITSCat(FITSCatalog):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def trim_redshift_range(self, compmin=0.5, zmin=None, zmax=None):
        '''
            - z cut
            - comp_BOSS cut
            - sector_SSR cut
        '''
        if zmin is None:
            zmin = self['Z'].min()-1.0e-7
        if zmax is None:
            zmax = self['Z'].max()+1.0e-7
        
        valid = (self['Z'] > zmin) & (self['Z'] < zmax)
        
        if 'IMATCH' in self.columns: # eBOSS or Legacy
            valid &= (self['IMATCH']==1) | (self['IMATCH']==2)
        if 'COMP_BOSS' in self.columns: 
            valid &= self['COMP_BOSS'] > compmin
        if 'sector_SSR' in self.columns:
            valid &= self['sector_SSR'] > compmin
        return self[valid]
        
    def sky_to_xyz(self, cosmo):
        self['Position'] = SkyToCartesian(self['RA'], self['DEC'], self['Z'], cosmo=cosmo)
        
    def apply_weight(self, use_systot=True):
        self['Weight'] = self['WEIGHT_NOZ'] * self['WEIGHT_CP']
        if use_systot:
            self['Weight'] *= self['WEIGHT_SYSTOT']
