import sys
import os
from lssutils.utils import split_NtoM
from sysnet.sources.train import forward
from sysnet.sources.models import DNNPoisson
from sysnet.sources.io import load_checkpoint, ImagingData, MyDataSet, DataLoader

from time import time
import fitsio as ft
import numpy as np
from astropy.table import Table
from glob import glob

from mpi4py import MPI


def chck2pid(chck):
    ch_ = chck.split('/')
    #print(ch_)
    return '_'.join([ch_[-2], ch_[-1].split('.')[0]])


def get_axes(maps):
    if maps == "sfd":
        axes=[0, 1, 2, 3, 4, 5, 6, 7,  10, 11, 12, 13]
    elif maps=="rongr":
        axes=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
    elif maps=="csfd":
        axes=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15]
    elif maps=="mud15":
        axes=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16]
    elif maps=="planck":
        axes=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 17]
    elif maps=="all":
        axes=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    else:
        raise ValueError(f'{maps} not known')
    return axes
 

def do_forward(checkpoints, rank, maps, oudir):

    axes = get_axes(maps) 
    nfolds = 5 
    num_features = len(axes) # nside=1024, 27 for nside=256
    nnstruct = (4, 20)

    model = DNNPoisson(*nnstruct, input_dim=num_features)
    for i, chck in enumerate(checkpoints):        

        t0 = time()
        checkpoint = load_checkpoint(chck, model)
        img_data = ImagingData(templates, checkpoint['stats'], axes=axes)
        dataloader = DataLoader(MyDataSet(img_data.x, img_data.y, img_data.p, img_data.w),
                                 batch_size=2000000,
                                 shuffle=False) # num_workers=4
                                
        if rank==0:print('finish data', time()-t0, i)
        result = forward(model, dataloader, {'device':'cpu'})        
        nnw = result[1].numpy().flatten()
        hpix = result[0].numpy()

        pid = chck2pid(chck)
        ouname = f'{oudir}/window_{pid}.fits'

        if rank==0:print('finish forward pass ', time()-t0, i)
        
        dt = Table([hpix, nnw], names=['hpix', 'weight'])
        dt.write(ouname, format='fits')
        if rank==0:print(f'save in {ouname}')
         

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    region = sys.argv[1]
    version = sys.argv[2]
    maps = sys.argv[3]
    print('region', region, maps)
    
    chcks = glob(f'/fs/ess/PHS0336/data/tanveer/dr9/{version}/elg_dnnp/{maps}_{region}_1024/model_*/snapshot_*.pth.tar')
    templates = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/{version}/nelg_features_{region}_1024.fits')    

   # nside=1024
    oudir = f'/fs/ess/PHS0336/data/tanveer/dr9/{version}/elg_dnnp/{maps}_{region}_1024/windows'
    if not os.path.exists(oudir):
        os.makedirs(oudir)
        print('rank 0 creates ', oudir)
else:
    chcks = None
    templates = None
    oudir = None
    maps = None

templates = comm.bcast(templates, root=0)
chcks = comm.bcast(chcks, root=0)
oudir = comm.bcast(oudir, root=0)
maps = comm.bcast(maps, root=0)

comm.Barrier()

my_i, my_f = split_NtoM(len(chcks), size, rank)
my_chcks = chcks[my_i:my_f+1]

if rank==0:print(rank, len(my_chcks), templates.size, my_chcks[:2])
do_forward(my_chcks, rank, maps, oudir)
