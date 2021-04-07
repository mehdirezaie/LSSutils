'''
    Code to make systematic maps given a ccd annotated file
    This calls some wrapper codes inside `desi_image_validation` developed by Marc Manera,
    which make use functions inside quicksip.py developed by Ashley J.Ross, Borris, ...
    
    Updates:
    Oct 11: deactivate airmass because DR7 is missing it    
    Jan 9: run for mjd_nobs for DR7
    python make_sysmaps.py --survey DECaLS --dr DR7 --localdir /Volumes/TimeMachine/data/DR7/sysmaps/
    Jan 10: run mjd for eBOSS chunks
    for chunk in 21 22 23 25;do python make_sysmaps.py --survey eBOSS --dr eboss${chunk} --localdir /Volumes/TimeMachine/data/ebo
ss/sysmaps/;echo $chunk is done;done
    Jan 11: run mjd for new updated version of eboss ccd files
for id in dr3 dr3_utah_ngc dr5-eboss dr5-eboss2; do python make_sysmaps.py --survey eBOSS --dr ${id} --localdir /Volumes/TimeMachine/data/eboss/sysmaps/;echo ${id} is done;done    
'''

import os
import lssutils.lab as make_maps
from argparse import ArgumentParser

ap = ArgumentParser(description='Systematic Maps Generating Routine')
ap.add_argument('--name', default='dr8maps')
ap.add_argument('-i', '--input_ccd', default='/home/mehdi/data/templates/ccds/ccds-annotated-dr8_combined.fits')
ap.add_argument('-o', '--out_dir', default='./')
ap.add_argument('-n', '--nside', default=256, type=int)
ap.add_argument('-b', '--bands', nargs='*', type=str, default=['r','g','z'])
ns = ap.parse_args()

locdir = os.path.dirname(ns.out_dir)
print(f'outputs will be under {locdir}')
if not os.path.exists(locdir):
    os.makedirs(locdir)
    print(f'created {locdir}')

make_maps(ns.input_ccd, ns.nside, ns.bands, ns.name, ns.out_dir)
