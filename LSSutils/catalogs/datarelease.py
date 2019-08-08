
cols_dr8 = ['ebv', 'loghi', 'nstar']\
         + ['depth_'+b+'_total' for b in 'rgz']\
         + ['fwhm_'+b+'_mean' for b in 'rgz']\
         + ['airmass_'+b+'_mean' for b in 'rgz']\
         + ['ccdskymag_'+b+'_mean' for b in 'rgz']\
         + ['exptime_'+b+'_total' for b in 'rgz']\
         + ['mjd_'+b+'_min' for b in 'rgz']

class Columns(object):
    def __init__(self, dr='dr8'):
        if dr=='dr8':
            self.cols = cols_dr8
        else:
            raise RuntimeError('%s not implemented'%dr)
        