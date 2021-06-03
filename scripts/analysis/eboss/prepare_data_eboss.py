""" Prepares the tabulated data for NN training

"""
import os
import logging
import pandas as pd
import fitsio as ft
import healpy as hp

from lssutils import setup_logging
from lssutils.utils import (EbossCat, HEALPixDataset, 
                            maps_eboss_v7p2, z_bins)

__zmin__ = 0.8
__zmax__ = 3.5


def prepare_table(config):
    """ prepare the tabulated data for nn regression
    """
    output_path_fn = lambda op, sl, ns:os.path.join(op, sl, f'{config.label}_eboss_{sl}_{ns}.fits')            

    nranbar = hp.nside2pixarea(config.nside, degrees=True)*5000  # 5000 randoms per sq deg

    templates = pd.read_hdf(config.templates_path, key='templates')  
    
    data = EbossCat(config.data_path, 
                    kind='data', zmin=__zmin__, zmax=__zmax__)
    
    randoms = EbossCat(config.randoms_path, 
                       kind='randoms', zmin=__zmin__, zmax=__zmax__)

    dataset = HEALPixDataset(data, randoms, templates, maps_eboss_v7p2)

    for slice_i in config.slices:
        
        if slice_i in z_bins:
            
            output_name = output_path_fn(config.output_path, slice_i, config.nside)
            zmin, zmax = z_bins[slice_i]            
            table = dataset.prepare(config.nside, zmin, zmax, label=config.label, nran_exp=nranbar)
            save_table(output_name, table)

            
def save_table(path, table):
    """ save the label features table as .fits file
    """
    path_dir = os.path.dirname(path)    
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        
    ft.write(path, table, clobber=True)
    
    
    
if __name__ == '__main__':
    
    from argparse import ArgumentParser
    
    ap = ArgumentParser(description='Prepare eBOSS data for NN regression')    
    ap.add_argument('-d', '--data_path', type=str, required=True)
    ap.add_argument('-r', '--randoms_path', type=str, required=True)
    ap.add_argument('-s', '--templates_path', type=str, required=True)
    ap.add_argument('-o', '--output_path', type=str, required=True)
    ap.add_argument('--label', type=str, default='ngal')
    ap.add_argument('-n', '--nside',  type=int, default=512)
    ap.add_argument('-sl', '--slices', type=str, nargs='*', default=['main', 'highz'])
    
    config = ap.parse_args()   
    
    
    setup_logging('info')        
    prepare_table(config)
