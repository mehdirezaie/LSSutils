
cols_dr8 = ['ebv', 'loghi', 'nstar']\
         + ['depth_'+b+'_total' for b in 'rgz']\
         + ['fwhm_'+b+'_mean' for b in 'rgz']\
         + ['airmass_'+b+'_mean' for b in 'rgz']\
         + ['ccdskymag_'+b+'_mean' for b in 'rgz']\
         + ['exptime_'+b+'_total' for b in 'rgz']\
         + ['mjd_'+b+'_min' for b in 'rgz']

cols_dr8_ts = ['GALDEPTH_G',
             'GALDEPTH_R',
             'GALDEPTH_Z',
             'PSFSIZE_G',
             'PSFSIZE_R',
             'PSFSIZE_Z',
             'EBV',
             'STARDENS']

cols_dr8_rand = ['STARDENS',
                 'EBV',
                 'PSFDEPTH_G',
                 'PSFDEPTH_R',
                 'PSFDEPTH_Z',
                 'GALDEPTH_G',
                 'GALDEPTH_R',
                 'GALDEPTH_Z',
                 'PSFDEPTH_W1',
                 'PSFDEPTH_W2',
                 'PSFSIZE_G',
                 'PSFSIZE_R',
                 'PSFSIZE_Z']



# w1 moon is removed
cols_eboss_qso_org = ['sky_g', 'sky_r', 'sky_i', 'sky_z', 
                     'depth_g', 'depth_r', 'depth_i','depth_z',
                     'psf_g','psf_r', 'psf_i', 'psf_z',
                     'w1_med', 'w1_covmed',
                     'star_density', 'ebv', 'airmass']


cols_eboss_v6_qso = ['sky_g', 'sky_r', 'sky_i', 'sky_z', 
                     'depth_g', 'depth_r', 'depth_i','depth_z',
                     'psf_g','psf_r', 'psf_i', 'psf_z',
                     'w1_med', 'w1_covmed',
                     'star_density', 'ebv', 'airmass', 'loghi', 'run']


cols_eboss_mocks_qso = ['depth_g_minus_ebv', 'star_density', 
                        'ebv',
                        'sky_g', 'sky_r', 'sky_i', 'sky_z',
                        'depth_g','depth_r', 'depth_i', 'depth_z', 
                        'psf_g', 'psf_r', 'psf_i', 'psf_z',
                        'w1_med', 'w1_covmed', 
                        'loghi', 'run', 'airmass']

cols_eboss_v6_qso_simp = ['sky_g', 'sky_r', 'sky_i', 'sky_z', 
                         'depth_g', 'depth_r', 'depth_i','depth_z',
                         'psf_g','psf_r', 'psf_i', 'psf_z',
                         'nstar', 'ebv', 'airmass', 'loghi', 'run']

cols_eboss_v7_qso = ['logSKY_G', 'logSKY_R', 'logSKY_I', 'logSKY_Z', 
                     'logDEPTH_G', 'logDEPTH_R', 'logDEPTH_I', 'logDEPTH_Z',
                     'PSF_G', 'PSF_R', 'PSF_I', 'PSF_Z', 
                     'W1_MED', 'logW1_COVMED', 'RUN', 'logAIRMASS',
                     'logEBV', 'log(1+STAR_DENSITY)', 'LOGHI']


def fixlabels(labels):
    labels_out = []
    
    for label in labels:
        a = label.split('_')
        if len(a)==2:
            b='-'.join([a[0], a[1]])
        elif len(a)==3:
            b='-'.join([a[0], a[1]])
        elif len(a)==1:
            b=a[0]
        else:
            raise ValueError("somthing is wrong")
            
        labels_out.append(b)
        
    return labels_out


class Columns(object):
    def __init__(self, dr='dr8'):
        if dr=='dr8':
            self.cols = cols_dr8
        if dr=='eboss_v6_qso':
            self.cols = cols_eboss_v6_qso
        else:
            raise RuntimeError('%s not implemented'%dr)
        
