
import os

from lssutils import setup_logging, CurrentMPIComm
from lssutils.lab import get_meandensity, maps_eboss_v7p2, EbossCat
from lssutils.utils import npix2nside

import pandas as pd
import numpy as np

@CurrentMPIComm.enable
def main(args, columns=maps_eboss_v7p2, comm=None):
    
    if comm.rank == 0:        
        # --- only rank 0
        zmin, zmax = args.zlim
        nside = args.nside
        use_systot = args.use_systot

        
        data_path = args.data_path
        randoms_path = args.randoms_path
        templates_path = args.templates_path
        output_path = args.output_path

        # read data, randoms, and templates
        data = EbossCat(data_path, kind='data', zmin=zmin, zmax=zmax)
        randoms = EbossCat(randoms_path, kind='randoms', zmin=zmin, zmax=zmax)

        templates = pd.read_hdf(templates_path, key='templates')
        sysm = templates[columns].values
        if nside is not None:
            assert sysm.shape[0] == 12*nside*nside, 'templates do not match with nside'
        else:
            nside = npix2nside(sysm.shape[0])
        
        # project to HEALPix
        if use_systot:
            ngal = data.to_hp(nside, zmin, zmax, raw=2)
            nran = randoms.to_hp(nside, zmin, zmax, raw=2)
        else:
            ngal = data.to_hp(nside, zmin, zmax, raw=1)
            nran = randoms.to_hp(nside, zmin, zmax, raw=1)

        # construct the mask    
        mask_nran = nran > 0
        mask_ngal = np.isfinite(ngal)
        mask_sysm = (~np.isfinite(sysm)).sum(axis=1) < 1

        mask = mask_sysm & mask_nran & mask_ngal  
    else:
        ngal = None
        nran = None
        mask = None
        sysm = None
        
        
    ngal = comm.bcast(ngal, root=0)
    nran = comm.bcast(nran, root=0)
    mask = comm.bcast(mask, root=0)
    sysm = comm.bcast(sysm, root=0)
    
    nnbar_list = get_meandensity(ngal, nran, mask, sysm, columns=columns)
    
    if comm.rank == 0:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            print(f'creating {output_dir}')
            os.makedirs(output_dir)
       
        np.save(output_path, nnbar_list)

if __name__ == '__main__':
    
    setup_logging("info") # turn on logging to screen
    comm = CurrentMPIComm.get()   
    
    if comm.rank == 0:
        print(f'hi from {comm.rank}')
        
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Mean Density')
        ap.add_argument('-d', '--data_path', required=True)
        ap.add_argument('-r', '--randoms_path', required=True)
        ap.add_argument('-t', '--templates_path', required=True)
        ap.add_argument('-o', '--output_path', required=True)
        ap.add_argument('-z', '--zlim', nargs='*', type=float, default=[0.8, 2.2]) 
        ap.add_argument('-n', '--nside', type=int, default=None)
        ap.add_argument('--use_systot', action='store_true')
        ns = ap.parse_args()

        for (key, value) in ns.__dict__.items():
            print(f'{key:15s} : {value}')                
    else:
        ns = None
        print(f'hey from {comm.rank}')
        
    main(ns)
            
            
        
    
