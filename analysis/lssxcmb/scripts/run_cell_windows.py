"""
	Calculates power spectrum of the windows
"""

import os
import fitsio as ft
import healpy as hp
import numpy as np

from time import time
from glob import glob
from lssutils import setup_logging, CurrentMPIComm
from lssutils.utils import split_NtoM, make_hp
from lssutils.stats.cl import AnaFast

nside = 1024
anafast = AnaFast()

@CurrentMPIComm.enable
def main(ns, comm=None):

    if comm.rank==0:
        for (key, value) in ns.__dict__.items():
            print(f'{key:15s} : {value}')
    
        # get list of windows
        windows = glob(os.path.join(ns.hpmap_dir, '*.fits'))
        print(f'# windows: {len(windows)}')

        # read imaging maps
        data = ft.read(ns.data_path)
        frac = make_hp(nside, data['hpix'], data['fracgood'])
        mask = make_hp(nside, data['hpix'], 1.0) > 0.5 
    else:
        windows = None
        frac    = None
        mask    = None

    windows = comm.bcast(windows, root=0)
    frac    = comm.bcast(frac, root=0)
    mask    = comm.bcast(mask, root=0)

    start, end = split_NtoM(len(windows), comm.size, comm.rank)
    cl = []
    for i in range(start, end+1):

        t0 = time()
        w_i = hp.read_map(windows[i], verbose=False, dtype=np.float64)
        w_i = w_i / np.median(w_i[mask])
        w_i = w_i.clip(0.5, 2.0)
        w_i = (w_i / w_i[mask].mean()) - 1.0
        cl_i = anafast(w_i, frac, mask)['cl'] 
        t1 = time()
        
        if comm.rank==0:print(f'rank-{comm.rank} finished {i} in {t1-t0} sec')
        cl.append(cl_i)

    comm.Barrier()
    cl = comm.gather(cl, root=0)
    
    if comm.rank==0:
        cl = np.array(cl)
        lmax = cl.shape[-1]
        cl = cl.reshape(-1, lmax)
        np.save(ns.output_path, cl)
    
if __name__ == '__main__':

    comm = CurrentMPIComm.get()   
 
    if comm.rank == 0:
        print(f'hi from {comm.rank}')
        
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Power Spectrum of the Windows')
        ap.add_argument('-d', '--data_path', required=True)
        ap.add_argument('-m', '--hpmap_dir', required=True)
        ap.add_argument('-o', '--output_path', required=True)
        ns = ap.parse_args()
    else:
        ns = None
        print(f'hey from {comm.rank}')
        
    main(ns)            

