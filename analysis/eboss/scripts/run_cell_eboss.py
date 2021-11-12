
import os

from lssutils import setup_logging, CurrentMPIComm
from lssutils.lab import get_cl, maps_eboss_v7p2, EbossCat
from lssutils.utils import nside2pixarea, npix2nside, ud_grade

import pandas as pd
import numpy as np

def angular_power(args, columns=maps_eboss_v7p2):
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
    if nside is None:
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
    
    if npix2nside(sysm.shape[0]) != nside:
        mask_sysm = ud_grade(mask_sysm, nside_out=nside)

    mask = mask_sysm & mask_nran & mask_ngal  

    nran_bar = nside2pixarea(nside, degrees=True)*5000.
    cls_list = get_cl(ngal, nran, mask, systematics=None, njack=0, nran_bar=nran_bar) # no systematics
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f'creating {output_dir}')
        os.makedirs(output_dir)

    np.save(output_path, cls_list)


    
    

@CurrentMPIComm.enable
def angular_power_wcross(args, columns=maps_eboss_v7p2, comm=None):
    
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
        ap.add_argument('-d', '--data_path', required=True)              # path to data
        ap.add_argument('-r', '--randoms_path', required=True)           # path to randoms
        ap.add_argument('-t', '--templates_path', required=True)         # path to imaging templates
        ap.add_argument('-o', '--output_path', required=True)            # path to output (.npy)
        ap.add_argument('-z', '--zlim', nargs='*', type=float, default=[0.8, 2.2]) # desired redshift cut
        ap.add_argument('-n', '--nside', type=int, default=None)         # healpix nside (if not given, it'll be taken from imaging maps)
        ap.add_argument('--use_systot', action='store_true')             # whether or not include imaging systot weights
        ap.add_argument('--auto_only', action='store_true')              # whether or not compute cross power between galaxy and imaging
        ap.add_argument('--cross_only', action='store_true')             # whether or not compute auto power of imaging
        ns = ap.parse_args()

        for (key, value) in ns.__dict__.items():
            print(f'{key:15s} : {value}')                
    else:
        ns = None
        print(f'hey from {comm.rank}')
    
    ns = comm.bcast(ns, root=0)
    
    if ns.auto_only:
        angular_power(ns)
    else:
        angular_power_wcross(ns)
            
            
        
    
