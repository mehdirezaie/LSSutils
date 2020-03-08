

import fitsio as ft
import numpy  as np
import pandas as pd
import healpy as hp
import logging

import sys
sys.path.append('/home/mehdi/github/LSSutils')
from LSSutils import setup_logging
from LSSutils.catalogs.combinefits import hd5_2_fits
from LSSutils.catalogs.datarelease import cols_dr8_rand
from LSSutils.utils import hpixsum



def main(mockid, my_cols=cols_dr8_rand):
    setup_logging('info')

    logger = logging.getLogger('RegressionPrep')

    # --- input parameters
    nside = 256    
    dataframe = '/home/mehdi/data/templates/pixweight-dr8-0.31.1.h5'
    random = f'/B/Shared/mehdi/mocksys/FA_EZmock_desi_ELG_v0_rand_00to2.hp{nside}.fits'
    output_dir = '/B/Shared/mehdi/mocksys/regression/'
    zcuts = {'low':[0.7, 1.0],
             'high':[1.0, 1.5],
             'all':[0.7, 1.5]}


    #---
    # start
    #---
    logger.info(f'ouput : {output_dir}')

    # --- templates
    df = pd.read_hdf(dataframe, key='templates')
    logger.info(f'read {dataframe}')

    # --- random
    hprandom = hp.read_map(random, verbose=False)
    logger.info(f'read {random}')

    # --- data
    data = ft.read(f'/B/Shared/Shadab/FA_LSS/FA_EZmock_desi_ELG_v0_{mockid}.fits')
    mask = ft.read(f'/B/Shared/Shadab/FA_LSS/EZmock_desi_v0.0_{mockid}/bool_index.fits')['bool_index']
    data = data[mask]
    z_rsd = data['Z_COSMO']+data['DZ_RSD']
    logger.info(f'read mock-{mockid}')


    for i, key_i in enumerate(zcuts):

        logger.info('split based on {}'.format(zcuts[key_i]))  


        # --- prepare the names for the output files
        hpcat     = None #output_dir + f'/galmap_{mockid}_{key_i}_{nside}.hp.fits'
        hpmask    = None #output_dir + f'/mask_{mockid}_{key_i}_{nside}.hp.fits'
        fracgood  = None #output_dir + f'/frac_{mockid}_{key_i}_{nside}.hp.fits'
        fitname   = None #output_dir + f'/ngal_features_{mockid}_{key_i}_{nside}.fits'    
        fitkfold  = output_dir + f'ngal_features_{mockid}_{key_i}_{nside}.5r.npy'

        good = (z_rsd >= zcuts[key_i][0]) & (z_rsd < zcuts[key_i][1])
        logger.info(f'total # : {good.sum()}')
        hpdata = hpixsum(nside, data[good]['RA'], data[good]['DEC'])


        # --- append the galaxy and random density
        dataframe_i = df.copy()
        dataframe_i['ngal'] = hpdata
        dataframe_i['nran'] = hprandom
        dataframe_i['nran'][hprandom == 0] = np.nan

        dataframe_i.replace()
        dataframe_i.replace([np.inf, -np.inf], value=np.nan, inplace=True) # replace inf
        dataframe_i.dropna(inplace=True)
        logger.info('df shape : {}'.format(dataframe_i.shape))
        logger.info('columns  : {}'.format(my_cols))

        for column in dataframe_i.columns:
            logger.info(f'{column}: {np.percentile(dataframe_i[column], [0,1,99, 100])}')

        # --- write 
        hd5_2_fits(dataframe_i, 
                      my_cols, 
                      fitname, 
                      hpmask, 
                      fracgood, 
                      fitkfold,
                      res=nside, 
                      k=5)

       
        
        
if __name__ == '__main__':
    mockid = int(sys.argv[1])
    #print(mockid)
    main(mockid)
