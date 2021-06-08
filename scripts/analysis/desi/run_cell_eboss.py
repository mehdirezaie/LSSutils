
import os

from lssutils import setup_logging, CurrentMPIComm
from lssutils.lab import get_cl, maps_desisv3
from lssutils.utils import nside2pixarea, npix2nside, ud_grade

import pandas as pd
import numpy as np


    
    

@CurrentMPIComm.enable
def angular_power_wcross(args, columns, comm=None):
    
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

        nran_bar = nside2pixarea(nside, degrees=True)*5000.

    else:
        ngal = None
        nran = None
        mask = None
        sysm = None
        nran_bar = None
        
        
    ngal = comm.bcast(ngal, root=0)
    nran = comm.bcast(nran, root=0)
    mask = comm.bcast(mask, root=0)
    sysm = comm.bcast(sysm, root=0)
    nran_bar = comm.bcast(nran_bar, root=0)
    
    cls_list = get_cl(ngal, nran, mask, systematics=sysm, njack=0, nran_bar=nran_bar, cross_only=args.cross_only)
    
    if comm.rank == 0:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            print(f'creating {output_dir}')
            os.makedirs(output_dir)
       
        np.save(output_path, cls_list)

if __name__ == '__main__':
    
    setup_logging("info") # turn on logging to screen
    comm = CurrentMPIComm.get()   
    
    if comm.rank == 0:
        print(f'hi from {comm.rank}')
        
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Angular Power Spectrum Cell')
        ap.add_argument('-d', '--data_path', required=True)
        ap.add_argument('-r', '--randoms_path', required=True)
        ap.add_argument('-t', '--templates_path', required=True)
        ap.add_argument('-o', '--output_path', required=True)
        ap.add_argument('--use_systot', action='store_true')
        ns = ap.parse_args()

        for (key, value) in ns.__dict__.items():
            print(f'{key:15s} : {value}')                
    else:
        ns = None
        print(f'hey from {comm.rank}')
    
    ns = comm.bcast(ns, root=0)
    angular_power_wcross(ns)
            
            
        
    
