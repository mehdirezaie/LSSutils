"""
	Calculates power spectrum of the Lognormal Mocks
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

nside = 256
anafast = AnaFast()

@CurrentMPIComm.enable
def main(ns, comm=None):

    if comm.rank==0:
        for (key, value) in ns.__dict__.items():
            print(f'{key:15s} : {value}')
    
        # get list of windows
        windows = glob(os.path.join(ns.hpmap_dir, f'{ns.fnl_tag}*.fits'))
        print(f'# windows: {len(windows)}')
        if len(windows) > 2:print(windows[:2])

        if ns.region in ['bmzls', 'ndecals', 'sdecals']:
            dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v3/nlrg_features_{ns.region}_256.fits')
            mask = np.zeros(12*nside*nside, '?')
            mask[dt['hpix']] = True
        else:
            mask = np.ones(12*nside*nside, '?')
        print(f'{ns.region} with fsky: {mask.mean()}')
    else:
        windows = None
        mask = None

    windows = comm.bcast(windows, root=0)
    mask = comm.bcast(mask, root=0)
    frac = mask*1.0 # complete mocks

    start, end = split_NtoM(len(windows), comm.size, comm.rank)
    cl = []
    for i in range(start, end+1):

        t0 = time()
        w_i = hp.read_map(windows[i], verbose=False, dtype=np.float64)
        w_i = (w_i / w_i[mask].mean()) - 1.0
        cl_i = anafast(w_i, frac, mask)['cl'] 
        t1 = time()
        
        if comm.rank==0:print(f'rank-{comm.rank} finished {i} in {t1-t0} sec')
        cl.append(cl_i)

    comm.Barrier()
    cl = comm.gather(cl, root=0)
    
    if comm.rank==0:
        cl = [cl_i for cl_i in cl if len(cl_i)!=0]
        cl = np.concatenate(cl)
        #print(cl.shape)

        output_dir = os.path.dirname(ns.output_path)
        if not os.path.exists(output_dir):
            print(f'creating {output_dir}')
            os.makedirs(output_dir)
        np.save(ns.output_path, cl, allow_pickle=False)
    
if __name__ == '__main__':

    comm = CurrentMPIComm.get()   
 
    if comm.rank == 0:
        print(f'hi from {comm.rank}')
        
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Power Spectrum of the Full Sky Lognormal Mocks')
        ap.add_argument('-m', '--hpmap_dir', required=True)
        ap.add_argument('-o', '--output_path', required=True)
        ap.add_argument('-t', '--fnl_tag', required=False, default='')
        ap.add_argument('-r', '--region', required=False, default='fullsky')
        ns = ap.parse_args()
    else:
        ns = None
        #print(f'hey from {comm.rank}')
        
    main(ns)            

