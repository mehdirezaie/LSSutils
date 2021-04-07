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
import lssutils.extrn.quicksip.qsdriver as qsdriver
from argparse import ArgumentParser

ap = ArgumentParser(description='systematic maps generating routine')
ap.add_argument('--survey', default='DECaLS')
ap.add_argument('--dr', default='DR5')
ap.add_argument('--localdir', default='/global/cscratch1/sd/mehdi/dr5_anand/sysmaps-v2/')
ap.add_argument('--nside', default=256, type=int)
ap.add_argument('--bands', nargs='*', type=str, default=['r','g','z'])
ns = ap.parse_args()

if not os.path.exists(ns.localdir):
    os.makedirs(ns.localdir)
    print(f'created {ns.localdir}')


verbose=False
for band in ns.bands:
    print('running band %s'%band)
    sample = qsdriver.mysample(ns.survey, ns.dr, band, ns.localdir, verbose, ns.nside)
    qsdriver.generate_maps(sample)
