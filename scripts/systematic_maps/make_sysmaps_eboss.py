""" Create systematics templates for eBOSS  """
import fitsio as ft
from lssutils.utils import make_sysmaps



lnHI = '/home/mehdi/data/templates/NHI_HPX.fits'
gaia = '/home/mehdi/data/templates/Gaia.dr2.bGT10.12g17.hp256.fits'

randoms = ft.read('/home/mehdi/data/templates/eBOSSrandoms.ran.fits', lower=True)
for nside in [256, 512]:

    path_output = f'/home/mehdi/data/templates/SDSS_WISE_HI_Gaia_imageprop_nside{nside}.h5'
    sysmaps = make_sysmaps(randoms, lnHI, gaia, nside)
    sysmaps.to_hdf(path_output, 'templates')

