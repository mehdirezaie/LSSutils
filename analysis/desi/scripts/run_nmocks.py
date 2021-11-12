
import os

from lssutils import setup_logging, CurrentMPIComm
from lssutils.lab import get_meandensity
from lssutils.utils import npix2nside, make_hp
from lssutils.utils import maps_dr9 as columns
import fitsio as ft
import numpy as np

@CurrentMPIComm.enable
def main(args, comm=None):
    
    if comm.rank == 0:       
        import healpy as hp
        # --- only rank 0        
        # read data, randoms, and templates
        data = ft.read(args.data_path)
        nside = 256
        
        #ngal = data['label']
        #nran = data['fracgood']
        nran = np.ones(data['hpix'].size)
        mask = np.ones(data['hpix'].size, '?')
        sysm = data['features']
        
        ngal_ = hp.read_map(args.hpmap_path, verbose=False)
        ngal = ngal_[data['hpix']]
        print(ngal.sum())
        
    else:
        ngal = None
        nran = None
        mask = None
        sysm = None
        selection_fn = None
        
       
        
        

if __name__ == '__main__':
    
    setup_logging("info") # turn on logging to screen
    comm = CurrentMPIComm.get()   
    
    if comm.rank == 0:
        #print(f'hi from {comm.rank}')
        
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Mean Density')
        ap.add_argument('-d', '--data_path', required=True)
        ap.add_argument('-m', '--hpmap_path', required=True)
        ap.add_argument('-o', '--output_path', required=True)
        ap.add_argument('-s', '--selection', default=None)
        ns = ap.parse_args()

        #for (key, value) in ns.__dict__.items():
        #    print(f'{key:15s} : {value}')                
    else:
        ns = None
        #print(f'hey from {comm.rank}')
        
    main(ns) 
