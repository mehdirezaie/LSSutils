
cols_dr8 = ['ebv', 'loghi', 'nstar']\
         + ['depth_'+b+'_total' for b in 'rgz']\
         + ['fwhm_'+b+'_mean' for b in 'rgz']\
         + ['airmass_'+b+'_mean' for b in 'rgz']\
         + ['ccdskymag_'+b+'_mean' for b in 'rgz']\
         + ['exptime_'+b+'_total' for b in 'rgz']\
         + ['mjd_'+b+'_min' for b in 'rgz']



cols_dr8_ts = ['galdepth_g', 'galdepth_r', 'galdepth_z', 
               'psfsize_g', 'psfsize_r', 'psfsize_z',
               'ebv', 'stardens']

# will rename the second ebv column
cols_dr8_ccdts = cols_dr8 + cols_dr8_ts
cols_dr8_ccdts[-2] = 'ebv2'

cols_dr8_rand = ['stardens', 'ebv', 
                 'psfdepth_g', 'psfdepth_r', 'psfdepth_z', 
                 'galdepth_g', 'galdepth_r', 'galdepth_z', 
                 'psfdepth_w1', 'psfdepth_w2', 
                 'psfsize_g', 'psfsize_r', 'psfsize_z']


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


def fixlabels(labels, addunit=True):
    columns = []
    for col in labels:

        # find unit
        if ('ebv' in col) | ('depth' in col):
            unit='[mag]'
        elif 'star' in col:
            unit=r'[deg$^{-2}$]'
        elif ('fwhm' in col) | ('psf' in col):
            unit='[arcsec]'
        elif 'airmass' in col:
            unit=''
        elif 'skymag' in col:
            unit=r'[mag/arcsec$^{2}$]'
        elif 'time' in col:
            unit='[sec]'
        elif 'mjd' in col:
            unit='[day]'
        elif 'hi' in col:
            col=r'log(HI/cm$^{2}$)' if addunit else 'logHI'
            unit=''
        else:
            raise RuntimeWarning(f'{col} not recognized')
            unit=''

        splits = col.split('_')

        if len(splits)>1:
            col = '-'.join([splits[0], splits[1]])
        else:
            col = splits[0]
        
        if addunit:
            col = ' '.join([col,unit])
            
        columns.append(col)
    return columns

class Columns(object):
    def __init__(self, dr='dr8'):
        if dr=='dr8':
            self.cols = cols_dr8
        if dr=='eboss_v6_qso':
            self.cols = cols_eboss_v6_qso
        else:
            raise RuntimeError('%s not implemented'%dr)
        
