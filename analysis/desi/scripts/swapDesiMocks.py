#!/usr/bin/env python
'''
    Code to read a galaxy, random catalog,


'''
import os
import logging


import sys
sys.path.append('/home/mehdi/github/lssutils')
from lssutils.catalogs.combinefits import SysWeight, DesiCatalog

def main(mockid=1,
         method='nn',
         model='plain',
         nside=256,
         zsplit='lowhigh',
         slices=['low', 'high'],
         target='ELG',
         version='v0',
         versiono='0.1'):    

    output_dir    = f'/B/Shared/mehdi/mocksys/{version}/{versiono}/'    
    data_name_in  = f'/B/Shared/Shadab/FA_LSS/FA_EZmock_desi_{target}_{version}_{mockid}.fits'
    bool_mask_in  = f'/B/Shared/Shadab/FA_LSS/EZmock_desi_{version}.0_{mockid}/bool_index.fits'

    tag           = '_'.join((version, versiono, method, model, zsplit))
    data_name_out = output_dir + f'FA_EZmock_desi_{target}_weights_{mockid}_{tag}.fits'

    # regression/results/2/all_256/regression/nn_plain/
    def weight(mockid, method, model, zcut, nside, output_dir):
        path2weights = output_dir +  f'regression/results/{mockid}/{zcut}_{nside}/regression/'
        method0 = 'nn' if method == 'nn' else 'mult'
        if method0 == 'mult':
           assert model == 'plain' 
        path2weights += f'{method0}_{model}/{method}-weights.hp{nside}.fits'
        return path2weights

#    zcuts = {'low':[[0.8, 1.5],   None],
#             'high':[[1.5, 2.2],  None],
#             'all':[[0.8, 2.2],   None],
#             'zhigh':[[2.2, 3.5], None],
#             'z1':[[0.8, 1.3], None],
#             'z2':[[1.3, 1.6], None],
#             'z3':[[1.6, 2.2], None]}
#
    zcuts = {'low':[[0.7, 1.0], None],
             'high':[[1.0, 1.5], None],
             'all':[[0.7, 1.5], None]}



    logger = logging.getLogger("Swapper")
    logger.info('results will be written under {}'.format(output_dir))  
    logger.info('swap the NN-z weights')
    logger.info(f'input data   : {data_name_in}')
    logger.info(f'bool_mask : {bool_mask_in}')
    logger.info(f'output data   : {data_name_out}')
    
    # --- check if the output directory exists
    if not os.path.isdir(output_dir):
        logger.info('create {}'.format(output_dir))
        #os.makedirs(output_dir)

    for zcut in zcuts:
        logger.info(f'zcut : {zcut}')
        weightname = weight(mockid, method, model, zcut, nside, output_dir)
        if not os.path.isfile(weightname):
            raise RuntimeError(f'{weightname} not found')
        zcuts[zcut][1]=SysWeight(weightname)


    #bool_index = ft.read(bool_mask_in)['bool_index'] # boolean mask

    data = DesiCatalog(data_name_in, bool_mask_in)
    data.swap(zcuts=zcuts, slices=slices)
    data.export_wsys(data_name_out)


    #data.make_plots(zcuts, slices=slices, filename=plotname)
    #data.to_fits(data_name_out)
    #random    = EbossCatalog(rand_name_in, zmin=zmin, zmax=zmax, kind='random')
    #newrandom = make_clustering_catalog_random(random.data, data.data)
    #newrandom.write(rand_name_out)    

    
    
if __name__ == '__main__':
        
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Prepare DESI Imaging Mocks')
    ap.add_argument('--model',   type=str, default='plain', help='eg:plain, other options are ablation and known ')
    ap.add_argument('--nside',   type=int, default=256, help='eg:256')
    ap.add_argument('--zsplit',  type=str, default='lowhigh', help='eg: lowhigh')
    ap.add_argument('--slices',  type=str, default=['low', 'high'], nargs='*', help="eg:['low', 'zhigh']")
    ap.add_argument('--method',  type=str, default='nn', help='nn, lin, or quad')
    ap.add_argument('--mockid', type=int, default=2)
    ap.add_argument('--target',  type=str, default='ELG', help='eg: ELG')
    ap.add_argument('--version', type=str, default='v0', help='eg: v0')
    ap.add_argument('--versiono',type=str, default='0.1', help='eg: 0.1')
    ns = ap.parse_args()    

    #--- default
    #
#    --model MODEL         eg:plain, other options are ablation and known
#    --nside NSIDE         eg:256
#    --zsplit ZSPLIT       eg: lowhigh
#    --slices [SLICES [SLICES ...]] eg:['low', 'zhigh']
#    --method METHOD       nn, lin, or quad
#    --mockid MOCKID
#    --target TARGET       eg: ELG
#    --version VERSION     eg: v0
#    --versiono VERSIONO   eg: 0.1
#
    from lssutils import setup_logging
    setup_logging('info')

    logger = logging.getLogger("Swapper")
    
    # 
    kwargs = ns.__dict__
    for (a,b) in zip(kwargs.keys(), kwargs.values()):
        logger.info('{:6s}{:15s} : {}'.format('', a, b))
        
    # -- call the function    
    main(**kwargs)
