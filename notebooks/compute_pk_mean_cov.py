
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import nbodykit.lab as nb


def get_pmeans(cap, iscont, method, nside):
    
    path = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
    pks = glob(f'{path}spectra_{cap}_{method}_mainhighz_{nside}_v7_{iscont}_*_main.json')
    
    nmocks = len(pks)
    print(nmocks)
    
    
    pk = []

    for pki in pks:
        
        dpk = nb.ConvolvedFFTPower.load(pki) 
        
        pk.append([dpk.poles['power_0'].real-dpk.attrs['shotnoise'], 
                   dpk.poles['power_2'].real])
        
        
    pk = np.array(pk)
    print(pk.shape)
    result = np.column_stack([dpk.poles['k'], dpk.poles['modes'], 
                             np.mean(pk[:, 0, :], axis=0), 
                             np.mean(pk[:, 1, :], axis=0)])
    
    return result


def get_covmax(cap, iscont, method, nside):
    
    path = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
    pks = glob(f'{path}spectra_{cap}_{method}_mainhighz_{nside}_v7_{iscont}_*_main.json')
    
    nmocks = len(pks)
    print(nmocks)
    
    
    pk = []

    for pki in pks:
        
        dpk = nb.ConvolvedFFTPower.load(pki) 
        
        pk.append([dpk.poles['power_0'].real-dpk.attrs['shotnoise'], 
                   dpk.poles['power_2'].real])
        
        
    pk = np.array(pk)
    nmocks = pk.shape[0]
    nbins = pk.shape[-1]
    hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
    
    #print(pk.shape)
    
    result = {'k':dpk.poles['k'], 
              'nmodes':dpk.poles['modes'], 
              'covp0':np.cov(pk[:, 0, :], rowvar=False)*hartlapf,
              'covp2':np.cov(pk[:, 1, :], rowvar=False)*hartlapf}
    
    return result

def get_covmax_wcorrect(cap, iscont, method, nside, slope, intercept):
    
    path = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
    pks = glob(f'{path}spectra_{cap}_{method}_mainhighz_{nside}_v7_{iscont}_*_main.json')
    
    nmocks = len(pks)
    print(nmocks)
    
    
    pk = []

    for pki in pks:
        
        dpk = nb.ConvolvedFFTPower.load(pki) 
        dpk_ = dpk.poles['power_0'].real-dpk.attrs['shotnoise']
        dpk_c = (1.-slope)*dpk_ - intercept
        pk.append(dpk_c)
        
        
    pk = np.array(pk)
    nmocks = pk.shape[0]
    nbins = pk.shape[-1]
    hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
    
    #print(pk.shape)
    
    result = {'k':dpk.poles['k'], 
              'nmodes':dpk.poles['modes'], 
              'covp0':np.cov(pk, rowvar=False)*hartlapf}
    
    return result


if __name__ == '__main__':
    
    # measure mean spectra of the mocks
    pks = {}

    methods = {'nnall':'all',
               'nnknown':'known',
               'standard':'knownsystot'}
    isconts = {'null':'0',
              'cont':'1'}
    caps = ['NGC', 'SGC']

    for cap in caps:
        for iscont in ['null', 'cont']:
            for method in ['nnknown', 'standard']: # 'nnall', 

                nsides = ['512'] #if method!='standard' else ['512'] # '256'

                for nside in nsides:

                    name = '_'.join([cap, iscont, method, nside])                
                    print(name)                

                    pk_ = get_pmeans(cap, isconts[iscont], methods[method], nside)                
                    pks[name] = pk_

    np.savez('./pk_ezmocks_ngcsgc_v1.0', **pks)                

    # measure cov. matrix spectra of the mocks
    covpks = {}

    for cap in caps:
        for iscont in ['null', 'cont']:
            for method in ['nnknown', 'standard']: # nnall

                nsides = ['512'] #, '256'] if method!='standard' else ['512']

                for nside in nsides:

                    name = '_'.join([cap, iscont, method, nside])                
                    #print(name)                

                    pk_ = get_covmax(cap, isconts[iscont], methods[method], nside)                
                    covpks[name] = pk_

    np.savez('./covpk_ezmocks_ngcsgc_v1.0', **covpks)