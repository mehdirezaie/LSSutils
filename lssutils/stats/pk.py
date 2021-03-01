
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
                          cosmology=None,
                          boxsize=None,
                          nmesh=512,
                          dk=0.002,
                          poles=[0, 2, 4],
                          kmax=None,
                          comm=None,
                          return_pk=False):
    
    if isinstance(nmesh, list):
        if len(nmesh)==1:
            nmesh=nmesh[0] 
        
    if isinstance(boxsize, list):
        if len(boxsize)==1:
            boxsize=boxsize[0]

        
        
    if (cosmology is None) or (cosmology=='boss'):
        # the fiducial BOSS DR12 cosmology 
        # see Alam et al. https://arxiv.org/abs/1607.03155        
        cosmo = Cosmology(h=0.676).match(Omega0_m=0.31)
        
    elif cosmology=='ezmock':
        _h = 0.6777
        _Ob0 = 0.048206
        _Ocdm0 = 0.307115 - _Ob0
        cosmo = Cosmology(h=_h, Omega_b=_Ob0, Omega_cdm=_Ocdm0,
                                       m_ncdm=None, n_s=0.9611, T_cmb=2.7255).match(sigma8=0.8225)
       
    elif cosmology=='boss2':
        h = 0.676
        redshift = 1.48
        cos = Cosmology(h=0.676,Omega0_b=0.022/h**2,n_s=0.97).match(Omega0_m=0.31)
        cosmo = cos.match(sigma8=0.8)

    elif isinstance(cosmology, Cosmology):
        cosmo = cosmology
        
    else:
        raise NotImplementedError(f'{cosmology} is not implemented')
        

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
    
    r = nb.ConvolvedFFTPower(mesh, poles=poles, dk=dk, kmin=0.0, kmax=kmax)
    
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
