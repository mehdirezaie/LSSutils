import matplotlib.pyplot as plt
import healpy as hp
from time import time

import sys
sys.path.insert(0, '/home/mehdi/github/sysnetdev')
sys.path.insert(0, '/home/mehdi/github/LSSutils/')
from sysnet.sources.train import forward
from sysnet.sources.models import DNN
from sysnet.sources.io import load_checkpoint, ImagingData, MyDataSet, DataLoader
from lssutils.utils import SysWeight, EbossCat, z_bins
from lssutils import setup_logging
from lssutils.stats.pk import run_ConvolvedFFTPower


import fitsio as ft
import numpy as np
from astropy.table import Table
from glob import glob
import nbodykit.lab as nb

class TrainedModel:

    def __init__(self, templates_path, metrics_path):
        """ initiate the model """
        self.data = ft.read(templates_path)
        self.stats = np.load(metrics_path, allow_pickle=True)['stats'].item()

    def run(self, chck_path, nfolds, axes=[0, 1, 5, 7, 13], boost_factors=[], nnstruct=(4, 20)):
        """ run """

        # boosting one feature
        data_new = self.data.copy()
        print(data_new['features'].shape)
        for ix, bf in boost_factors:
            if ix not in axes:raise ValueError('axis not included')
            #print(ix, bf)
            data_new['features'][:, ix] *= bf
            #print(ix, bf, data_new['features'][:5, ix], self.data['features'][:5, ix])

        num_features = len(axes)
        model = DNN(*nnstruct, input_dim=num_features)

        nnw = []
        hpix = None

        for p in range(nfolds):

            img_data = ImagingData(data_new, self.stats[p], axes=axes)
            dataloader = DataLoader(MyDataSet(img_data.x, img_data.y, img_data.p, img_data.w),
                                     batch_size=100000,
                                     shuffle=False,
                                     num_workers=4)

            chcks = glob(f'{chck_path}/model_{p}_*/best.pth.tar')
            print(chcks[:2], len(chcks), p)

            for chck in chcks:

                checkpoint = load_checkpoint(chck, model)
                result = forward(model, dataloader, {'device':'cpu'})
                if hpix is None:
                    hpix = result[0].numpy()

                nnw.append(result[1].numpy().flatten())
                print('.', end='')


        dt = Table([hpix, np.array(nnw).T], names=['hpix', 'weight'])
        return dt
        # dt.write('/home/mehdi/data/tanveer/elg_mse_snapshots/nn-weights-combined.fits', format='fits')


class NNWeight(SysWeight):

    def __init__(self, wnn, nside, fix=True, clip=False):

        wnn_hp = np.zeros(12*nside*nside)
        wnn_hp[wnn['hpix']] = wnn['weight'].mean(axis=1)

        super(NNWeight, self).__init__(wnn_hp, ismap=True, fix=fix, clip=clip)

if __name__ == '__main__':
    setup_logging('info')
    templates_path = '/home/mehdi/data/eboss/data/v7_2/3.0/NGC/512/main/ngal_eboss_main_512.fits'
    metrics_path = '/home/mehdi/data/eboss/data/v7_2/3.0/NGC/512/main/nn_pnll_known/metrics.npz'
    chck_path = '/home/mehdi/data/eboss/data/v7_2/3.0/NGC/512/main/nn_pnll_known'


    sys_axes = {'nstar':0, 
                'ebv':1, 
                'skyi':5, 
                'depthg':7, 
                'psfi':13}

    tm = TrainedModel(templates_path, metrics_path)


    # read data, randoms, and prepare mappers
    dat = EbossCat('/home/mehdi/data/eboss/data/v7_2/eBOSS_QSO_full_NGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    ran = EbossCat('/home/mehdi/data/eboss/data/v7_2/eBOSS_QSO_full_NGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)

    for sys_name, sys_axis in sys_axes.items():
        print(sys_name, sys_axis)
        for bf in [2., 2.5, 3., 3.5, 4.]:
            p = '/home/mehdi/data/eboss/data/v7_2/3.0/catalogs_boosting/'
            dat_name = f'{p}eBOSS_QSO_known_{sys_name}_{bf:.1f}_NGC_v7_2.dat.fits'
            out_name = dat_name.replace('.dat.fits', '.pk.json')
            ran_name = dat_name.replace('.dat.', '.ran.')
            print(dat_name, out_name)

            t0 = time()
            dt_0 = tm.run(chck_path, 5, boost_factors=[(sys_axis, bf)])
            t1 = time()
            print('forward pass', t1-t0)

            nnwmap = {'main':(z_bins['main'], NNWeight(dt_0, 512))}
            dat.swap(nnwmap)
            ran.reassign_zattrs(dat)
           
            dat.to_fits(dat_name)
            ran.to_fits(ran_name)
            run_ConvolvedFFTPower(dat_name, ran_name, out_name, zmin=0.8, zmax=2.2, 
                                  dk=0.01, boxsize=6600., return_pk=True, poles=[0])
