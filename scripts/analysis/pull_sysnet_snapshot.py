import sys
sys.path.insert(0, '/home/mehdi/github/sysnetdev')
from sysnet.sources.train import forward
from sysnet.sources.models import DNN
from sysnet.sources.io import load_checkpoint, ImagingData, MyDataSet, DataLoader


import fitsio as ft
import numpy as np
from astropy.table import Table
from glob import glob

templates = ft.read(f'/home/mehdi/data/tanveer/dr8_elg_0.32.0_256.fits')


nfolds = 5 
num_features = 27
nnstruct = (4, 20)


nnw = []

hpix = None
    
metrics = np.load(f'/home/mehdi/data/tanveer/elg_mse_snapshots/metrics.npz', allow_pickle=True)
stats = metrics['stats'].item()
    
for p in range(nfolds):

    img_data = ImagingData(templates, stats[p])
    dataloader = DataLoader(MyDataSet(img_data.x, img_data.y, img_data.p, img_data.w),
                             batch_size=4098,
                             shuffle=False,
                             num_workers=0)

    chcks = glob(f'/home/mehdi/data/tanveer/elg_mse_snapshots/model_{p}_*/snapshot_*.pth.tar')
    #print(chcks[:2], len(chcks), p)
    for chck in chcks:        
        model = DNN(*nnstruct, input_dim=num_features)
        checkpoint = load_checkpoint(chck, model)
        result = forward(model, dataloader, {'device':'cpu'})        


        if hpix is None:
            hpix = result[0].numpy()    

        nnw.append(result[1].numpy().flatten())
        print('.', end='')        
        
        
dt = Table([hpix, np.array(nnw).T], names=['hpix', 'weight'])
dt.write('/home/mehdi/data/tanveer/elg_mse_snapshots/nn-weights-combined.fits', format='fits')