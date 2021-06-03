""" Pull Sysnet Jackknifes together 
"""

import sys
sys.path.insert(0, '/home/mehdi/github/sysnetdev')
from sysnet.sources.train import forward
from sysnet.sources.models import DNN
from sysnet.sources.io import load_checkpoint, ImagingData, MyDataSet, DataLoader


import fitsio as ft
import numpy as np
from astropy.table import Table



templates = ft.read(f'/home/mehdi/data/tanveer/dr8_elg_0.32.0_256.fits')

njacks = 25
nfolds = 5 
num_features = 27
nnstruct = (4, 20)
seed = 2664485226

nnw = []

hpix = None
for j in range(njacks):
    
    metrics = np.load(f'/home/mehdi/data/tanveer/elg_mse_jk/jk{j}/metrics.npz', allow_pickle=True)
    stats = metrics['stats'].item()
    
    for p in range(nfolds):
        
        img_data = ImagingData(templates, stats[p])
        dataloader = DataLoader(MyDataSet(img_data.x, img_data.y, img_data.p, img_data.w),
                                 batch_size=4098,
                                 shuffle=False,
                                 num_workers=0)
        
        chck = f'/home/mehdi/data/tanveer/elg_mse_jk/jk{j}/model_{p}_{seed}/best.pth.tar'
        
        model = DNN(*nnstruct, input_dim=num_features)
        checkpoint = load_checkpoint(chck, model)
        result = forward(model, dataloader, {'device':'cpu'})        

        
        if hpix is None:
            hpix = result[0].numpy()    
        
        nnw.append(result[1].numpy().flatten())
        print('.', end='')        
        
        
dt = Table([hpix, np.array(nnw).T], names=['hpix', 'weight'])
dt.write('/home/mehdi/data/tanveer/elg_mse_jk/nn-weights-combined.fits', format='fits')
