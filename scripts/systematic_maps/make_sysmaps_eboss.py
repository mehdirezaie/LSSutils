
import sys
import fitsio as ft

from lssutils.utils import make_sysmaps


nside = int(sys.argv[1])

randoms_path = '/home/mehdi/data/templates/eBOSSrandoms.ran.fits'
nhi_path = '/home/mehdi/data/templates/NHI_HPX.fits'
nstar_path = '/home/mehdi/data/templates/allstars17.519.9Healpixall256.dat'
templates_path = f'/home/mehdi/data/templates/SDSS_WISE_HI_imageprop_nside{nside}.h5'

randoms = ft.read(randoms_path, lower=True)
sysmaps = make_sysmaps(randoms, nhi_path, nstar_path, nside)
sysmaps.to_hdf(templates_path, 'templates')