
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
        # --- only rank 0        
        # read data, randoms, and templates
        data = ft.read(args.data_path)
        nside = 256
        
        ngal = data['label']
        nran = data['fracgood']
        mask = np.ones_like(ngal, '?')
        sysm = data['features']
        
        if args.selection is not None:
            s_ = ft.read(args.selection)           
            selection_fn = make_hp(256, s_['hpix'], np.median(s_['weight'], axis=1))#.mean(axis=1))
            selection_fn = selection_fn[data['hpix']]
            print(np.percentile(selection_fn[mask], [0, 1, 99, 100]))

        else:
            selection_fn = None
        
    else:
        ngal = None
        nran = None
        mask = None
        sysm = None
        selection_fn = None
        
        
        
    ngal = comm.bcast(ngal, root=0)
    nran = comm.bcast(nran, root=0)
    mask = comm.bcast(mask, root=0)
    sysm = comm.bcast(sysm, root=0)
    selection_fn = comm.bcast(selection_fn, root=0)
    
    nnbar_list = get_meandensity(ngal, nran, mask, sysm, 
                                 columns=columns, selection_fn=selection_fn, binning='simple', percentiles=[1, 99],
                                 global_nbar=True)
    
    if comm.rank == 0:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            print(f'creating {output_dir}')
            os.makedirs(output_dir)
       
        np.save(args.output_path, nnbar_list)
        
        
        

if __name__ == '__main__':
    
    setup_logging("info") # turn on logging to screen
    comm = CurrentMPIComm.get()   
    
    if comm.rank == 0:
        print(f'hi from {comm.rank}')
        
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Mean Density')
        ap.add_argument('-d', '--data_path', required=True)
        ap.add_argument('-o', '--output_path', required=True)
        ap.add_argument('-s', '--selection', default=None)
        ns = ap.parse_args()

        for (key, value) in ns.__dict__.items():
            print(f'{key:15s} : {value}')                
    else:
        ns = None
        print(f'hey from {comm.rank}')
        
    main(ns)    
