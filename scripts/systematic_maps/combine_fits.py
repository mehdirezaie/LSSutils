'''
    Code to read all imaging maps 
    and write them onto a hdf file
'''

import os
from glob import glob
from lssutils.lab import combine_fits

#--- inputs 
from argparse import ArgumentParser
ap = ArgumentParser(description='systematic maps combining routine')
ap.add_argument('-n', '--name', type=str, default='dr8m')
ap.add_argument('--nside',  type=int,  default=1024)
ap.add_argument('-i', '--in_dir',  type=str,  default='/home/mehdi/data/templates/dr8/')
ap.add_argument('-o', '--out_dir',  type=str,  default='/home/mehdi/data/templates/dr8/')
ap.add_argument('--add_galactic', action='store_true')
ns = ap.parse_args()


in_path = os.path.join(ns.in_dir, ns.name, 
                       f'nside{ns.nside}_oversamp1', f'{ns.name}*.fits.gz')
in_paths = glob(in_path)

out_path = os.path.join(ns.out_dir, f'{ns.name}_nside{ns.nside}.h5')

combine_fits(in_paths, ns.nside, add_galactic=ns.add_galactic, write_to=out_path)
print(f'wrote {out_path}')