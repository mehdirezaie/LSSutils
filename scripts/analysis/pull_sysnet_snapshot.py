import sys
sys.path.insert(0, '/home/mehdi/github/sysnetdev')
from sysnet.sources.train import forward
from sysnet.sources.models import DNN
from sysnet.sources.io import load_checkpoint, ImagingData, MyDataSet, DataLoader

from time import time
import fitsio as ft
import numpy as np
from astropy.table import Table
from glob import glob

#templates = ft.read(f'/home/mehdi/data/tanveer/dr8_elg_0.32.0_256.fits')
templates = ft.read(f'/home/mehdi/data/tanveer/dr8_elg_ccd_1024.fits')   # nside=1024
print(templates.size)

nfolds = 5 
num_features = 21 # nside=1024, 27 for nside=256
nnstruct = (3, 20)

model = DNN(*nnstruct, input_dim=num_features)

nnw = []

chcks = glob(f'/home/mehdi/data/tanveer/elg_mse_snapshots/model_*/snapshot_*.pth.tar')
print(chcks[:2], len(chcks))

for i, chck in enumerate(chcks):        
    t0 = time()
    checkpoint = load_checkpoint(chck, model)
    img_data = ImagingData(templates, checkpoint['stats'])
    dataloader = DataLoader(MyDataSet(img_data.x, img_data.y, img_data.p, img_data.w),
                             batch_size=1000000,
                             shuffle=False,
                             num_workers=2)
                            
    print('finish data', time()-t0, i)
    result = forward(model, dataloader, {'device':'cpu'})        
    nnw.append(result[1].numpy().flatten())
    print('finish forward pass ', time()-t0, i)
     
hpix = result[0].numpy()           
dt = Table([hpix, np.array(nnw).T], names=['hpix', 'weight'])
dt.write('/home/mehdi/data/tanveer/elg_mse_snapshots/nn-weights-combined.fits', format='fits')
