import sys
import os
import logging

import numpy as np
import pandas as pd
import healpy as hp
import fitsio as ft

sys.path.insert(0, '/users/PHS0336/medirz90/github/LSSutils')
from lssutils.stats.cl import get_cl
sys.path.insert(0, '/users/PHS0336/medirz90/github/regressis')
from regressis import PhotometricDataFrame, Regression, DR9Footprint
from regressis.utils import setup_logging


logger = logging.getLogger('MockTest')
setup_logging()

path2input = sys.argv[1]
path2output = sys.argv[2]
print(f'input: {path2input}')
print(f'output: {path2output}')


version, tracer, suffix_tracer = 'SV3', 'LRG', 'mock'
dr9_footprint = DR9Footprint(256, mask_lmc=False, clear_south=False, mask_around_des=False, cut_desi=False)
params = dict()
params['output_dir'] = None
params['use_median'] = False
params['use_new_norm'] = False
params['regions'] = ['North']
dataframe = PhotometricDataFrame(version, tracer, dr9_footprint, suffix_tracer, **params)

dt = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_bmzls_256.fits')
ng = hp.read_map(path2input)

r2n = hp.reorder(np.arange(12*256*256), r2n=True)
feat_ = dt['features'][:, [0, 1, 4, 6, 11]]
featr_ = np.zeros((12*256*256, 5))
featr_[dt['hpix']] = feat_
features = featr_[r2n]
targets = ng[r2n]
fracarea = np.zeros(12*256*256)
fracarea_ = np.ones(targets.size)*np.nan
fracarea_[dt['hpix']] = 1.0
fracarea = fracarea_[r2n]

feature_names = ['ebv', 'nstar', 'galdepth_z', 'psfdepth_g', 'psfsize_g']
featpd = pd.DataFrame(features, columns=feature_names)

logger.info('Features')
dataframe.set_features(featpd, sel_columns=feature_names, 
                       use_sgr_stream=False, features_toplot=False) 
logger.info('Targets')
dataframe.set_targets(targets, fracarea=fracarea, )
logger.info('Build')
dataframe.build(cut_fracarea=False)

feature_names = ['ebv', 'nstar', 'galdepth_z', 'psfdepth_g', 'psfsize_g']
use_kfold = True
regressor_params = None
nfold_params = {'North':6}
regression = Regression(dataframe, feature_names=feature_names,
                        regressor_params=regressor_params, nfold_params=nfold_params,
                        regressor='RF', suffix_regressor='', use_kfold=use_kfold,
                        n_jobs=1, seed=123, compute_permutation_importance=False, overwrite=True)
wsys = 1./regression.get_weight(save=False).map


# measure C_ells
mask = fracarea_ > 0
cl_before = get_cl(ng, fracarea_, mask)
cl_after = get_cl(ng, fracarea_, mask, selection_fn=hp.reorder(wsys, n2r=True))




np.savez(path2output, **{'cl_before':cl_before['cl_gg']['cl'], 'cl_after':cl_after['cl_gg']['cl']})
