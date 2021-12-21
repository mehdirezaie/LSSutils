import sys
#sys.path.insert(0, '/home/mehdi/github/sysnetdev')
#sys.path.insert(0, '/home/mehdi/github/LSSutils')

from lssutils.utils import split_NtoM
from sysnet.sources.train import forward
from sysnet.sources.models import DNN
from sysnet.sources.io import load_checkpoint, ImagingData, MyDataSet, DataLoader

from time import time
import fitsio as ft
import numpy as np
from astropy.table import Table
from glob import glob

from mpi4py import MPI


def do_forward(checkpoints, rank):

    nfolds = 5 
    num_features = 21 # nside=1024, 27 for nside=256
    nnstruct = (3, 20)

    model = DNN(*nnstruct, input_dim=num_features)
    for i, chck in enumerate(checkpoints):        

        t0 = time()
        checkpoint = load_checkpoint(chck, model)
        img_data = ImagingData(templates, checkpoint['stats'])
        dataloader = DataLoader(MyDataSet(img_data.x, img_data.y, img_data.p, img_data.w),
                                 batch_size=2000000,
                                 shuffle=False) # num_workers=4
                                
        if rank==0:print('finish data', time()-t0, i)
        result = forward(model, dataloader, {'device':'cpu'})        
        nnw = result[1].numpy().flatten()
        hpix = result[0].numpy()

        pid = chck.split('/')[-2].split('_')[1]+'_'+chck.split('/')[-1].split('.')[0]
        ouname = f'/home/mehdi/data/tanveer/elg_mse_snapshots/windows/window_{pid}.fits'
        if rank==0:print('finish forward pass ', time()-t0, i)
        
        dt = Table([hpix, nnw], names=['hpix', 'weight'])
        dt.write(ouname, format='fits')
        if rank==0:print(f'save in {ouname}')
         

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    chcks = glob(f'/home/mehdi/data/tanveer/elg_mse_snapshots/model_*/snapshot_*.pth.tar')
    #templates = ft.read(f'/home/mehdi/data/tanveer/dr8_elg_0.32.0_256.fits')
    templates = ft.read(f'/home/mehdi/data/tanveer/dr8_elg_ccd_1024.fits')   # nside=1024
else:
    chcks = None
    templates = None

templates = comm.bcast(templates, root=0)
chcks = comm.bcast(chcks, root=0)


my_i, my_f = split_NtoM(len(chcks), size, rank)
my_chcks = chcks[my_i:my_f+1]

if rank==0:print(rank, len(my_chcks), templates.size, my_chcks[:2])
do_forward(my_chcks, rank)
