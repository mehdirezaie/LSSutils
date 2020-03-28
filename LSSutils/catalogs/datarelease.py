'''
Imaging templates for different datasets

'''
zcuts = {'low':[0.8, 1.5],
         'high':[1.5, 2.2],
         'all':[0.8, 2.2],
         'zhigh':[2.2, 3.5],
         'z1':[0.8, 1.3],
         'z2':[1.3, 1.6],
         'z3':[1.6, 2.2]}

cols_dr8_ccd = ['ebv', 'loghi', 'nstar']\
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
cols_dr8_ccdts = cols_dr8_ccd + cols_dr8_ts
cols_dr8_ccdts[-2] = 'ebv*'

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
    '''
    The fixlabels function.
    
    The function fixes labels
    
    Parameters
    ----------
    labels : list
        1-D list of strings that are the names of the attributes
    
    addunit : boolean, optional
        Boolean argument to add the units to each label
        
    Returns
    -------
    labels_fixed : float
        1-D list of strings that are the fixed names of the attributes.
        
    Examples    
    --------
    >>> columns = ['ebv', 'depth_r_max', 'loghi']
    >>> lab.catalogs.datarelease.fixlabels(columns, addunit=False)
    ['ebv', 'depth-r', 'logHI']

    >>> lab.catalogs.datarelease.fixlabels(columns, addunit=True)
    ['ebv [mag]', 'depth-r [mag]', 'log(HI/cm$^{2}$) ']
        
    '''
    units = {0:'[mag]',
              1:r'[deg$^{-2}$]',
              2:'[arcsec]',
              3:r'[mag/arcsec$^{2}$]',
              4:'[sec]',
              5:'[day]',
              6:''}

    labels_fixed = []
    for col in labels:

        # find unit
        if ('ebv' in col) | ('depth' in col):
            unit=units[0]
            
        elif 'star' in col:
            unit=units[1]
            
        elif ('fwhm' in col) | ('psf' in col):
            unit=units[2]
            
        elif 'airmass' in col:
            unit=units[6]
            
        elif 'skymag' in col:
            if 'ccd' in col:
                col = col.replace('ccd', '')
            unit=units[3]
            
        elif 'time' in col:
            unit=units[4]
            
        elif 'mjd' in col:
            unit=units[5]
            
        elif 'hi' in col:
            col=r'log(HI/cm$^{2}$)' if addunit else 'logHI'
            unit=units[6]
            
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
            
        labels_fixed.append(col)
        
    return labels_fixed