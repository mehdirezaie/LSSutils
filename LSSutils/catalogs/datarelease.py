
cols_dr8 = ['ebv', 'loghi', 'nstar']\
         + ['depth_'+b+'_total' for b in 'rgz']\
         + ['fwhm_'+b+'_mean' for b in 'rgz']\
         + ['airmass_'+b+'_mean' for b in 'rgz']\
         + ['ccdskymag_'+b+'_mean' for b in 'rgz']\
         + ['exptime_'+b+'_total' for b in 'rgz']\
         + ['mjd_'+b+'_min' for b in 'rgz']


# w1 moon is removed
cols_eboss_v6_qso = ['sky_g', 'sky_r', 'sky_i', 'sky_z', 
                     'depth_g', 'depth_r', 'depth_i','depth_z',
                     'psf_g','psf_r', 'psf_i', 'psf_z',
                     'w1_med', 'w1_covmed',
                     'star_density', 'ebv', 'airmass', 'loghi', 'run']


class Columns(object):
    def __init__(self, dr='dr8'):
        if dr=='dr8':
            self.cols = cols_dr8
        if dr=='eboss_v6_qso':
            self.cols = cols_eboss_v6_qso
        else:
            raise RuntimeError('%s not implemented'%dr)
        
