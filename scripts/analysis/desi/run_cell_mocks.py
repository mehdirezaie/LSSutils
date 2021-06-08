
import sys
import os
import numpy as np
import healpy as hp
import pandas as pd
from lssutils.lab import AnaFast, maps_dr9sv3, make_overdensity

def get_cl(delta, weight, mask, systematics):
    af_kws = dict(njack=0)
    af = AnaFast()

    #--- auto power spectrum
    cl_gg = af(delta, weight, mask, **af_kws)

    cl_sg_list = []
    for i in range(systematics.shape[1]):
        systematic_i = make_overdensity(systematics[:, i],
                                        weight, mask, is_sys=True)
        cl_sg_list.append(af(delta, weight, mask,
                        map2=systematic_i, weight2=weight, mask2=mask, **af_kws))


    output = {
        'cl_gg':cl_gg,
        'cl_sg':cl_sg_list,
    }
    return output

 
i = int(sys.argv[1])


mock_ = lambda i:f'/home/mehdi/data/dr9v0.57.0/p38fnl/data/mocks/{i:03d}.hp.fits'# sys.argv[1]
cl_ = lambda i:f'/home/mehdi/data/dr9v0.57.0/p38fnl/data/mocks/cl/cl_{i:03d}.npz'# sys.argv[1]
mask_ = '/home/mehdi/data/dr9v0.57.0/p38fnl/data/mask_bmzls.hp.fits'
templ_ = '/home/mehdi/data/templates/dr9/pixweight_dark_dr9m_nside256.h5'


delta = hp.read_map(mock_(i))
output_path = cl_(i)


#--- read templates
columns = maps_dr9sv3
templ = pd.read_hdf(templ_)
templ_np = templ[columns].values

# --- read mask
mask = hp.read_map(mask_) > 0.5
mask_sysm = (~np.isfinite(templ_np)).sum(axis=1) < 1
mask &= mask_sysm

weight = mask.astype('float64')


cls_list = get_cl(delta, weight, mask, systematics=templ_np)
print(cls_list)

output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    print(f'creating {output_dir}')
    os.makedirs(output_dir)

np.savez(output_path, **cls_list)
