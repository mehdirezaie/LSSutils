'''
	This code is written by Marc Manera
	However its functionality has been reduced to 
	only generating the photometric maps from ccd files
	mysample is a class that facilitates reading the input
	parameters. The main task is done inside project_and_write_maps
'''
import numpy as np
import astropy.io.fits as pyfits
import sys
from .quicksip import project_and_write_maps_simp



### ------------ A couple of useful conversions -----------------------

def Magtonanomaggies(m):
	return 10.**(-m/2.5+9.)
	#-2.5 * (log(nm,10.) - 9.)

### ------------ SHARED CLASS: HARDCODED INPUTS GO HERE ------------------------
###    Please, add here your own harcoded values if any, so other may use them 

class mysample(object):
    """
    (c) Marc Manera
    This class mantains the basic information of the sample
    to minimize hardcoded parameters in the test functions

    modified by MR Aug 7, 2019

    Everyone is meant to call mysample to obtain information like 
         - path to ccd-annotated files   : ccds
         - extintion coefficient         : extc
    """                                  
    def __init__(self, survey, DR, band, localdir, verb, nside):
        """ 
        Initialize image survey, data release, band, output path
        Calculate variables and paths
        """   
        self.survey     = survey
        self.DR         = DR
        self.band       = band
        self.localdir   = localdir 
        self.verbose    = verb
        self.nside      = nside

        # Check bands
        if self.band not in ['r', 'g', 'z']: 
            raise RuntimeError("Band seems wrong! options are 'g' 'r' 'z'")        
              
        # Check surveys
        if self.survey not in ['DECaLS', 'BASS', 'MZLS', 'eBOSS']:
            raise RuntimeError("Survey seems wrong options are 'DECAaLS' 'BASS' MZLS' 'eBOSS' ")

        # Annotated CCD paths  
        if(self.DR == 'DR3'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr3/'
            #self.ccds =inputdir+'ccds-annotated-decals.fits.gz'
            self.ccds ='/global/project/projectdirs/desi/users/mehdi/trunk/'\
                      +'dr3-ccd-annotated-nondecals-extra-decals.fits' # to include all 
            self.catalog = 'DECaLS_DR3'
            if(self.survey != 'DECaLS'): raise RuntimeError("Survey name seems inconsistent")
        elif(self.DR == 'DR4'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr4/'
            if (band == 'g' or band == 'r'):
                #self.ccds = inputdir+'ccds-annotated-dr4-90prime.fits.gz'
                self.ccds = inputdir+'ccds-annotated-bass.fits.gz'
                self.catalog = 'BASS_DR4'
                if(self.survey != 'BASS'): raise RuntimeError("Survey name seems inconsistent")

            elif(band == 'z'):
                #self.ccds = inputdir+'ccds-annotated-dr4-mzls.fits.gz'
                self.ccds = inputdir+'ccds-annotated-mzls.fits.gz'
                self.catalog = 'MZLS_DR4'
                if(self.survey != 'MZLS'): raise RuntimeError("Survey name seems inconsistent")
            else: raise RuntimeError("Input sample band seems inconsisent")
        elif(self.DR == 'DR5'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr5/'
            self.ccds =inputdir+'ccds-annotated-dr5.fits.gz'
            self.catalog = 'DECaLS_DR5'
            if(self.survey != 'DECaLS'): raise RuntimeError("Survey name seems inconsistent")
        elif (self.DR == 'DR7'): 
            inputdir = '/Volumes/TimeMachine/data/DR7/'
            self.ccds = inputdir+'ccds-annotated-dr7.fits.gz'
            self.catalog = 'DECaLS_DR7'
        elif (self.DR in ['eboss21', 'eboss22', 'eboss23', 'eboss25']): # Jan 10, 2019 for eBOSS chunks
            inputdir     =  '/Volumes/TimeMachine/data/eboss/sysmaps/ccdfiles/'
            #self.ccds    =  inputdir + 'survey-ccds.'+self.DR+'.fits.gz'
            self.ccds    =  inputdir + 'ccds-annotated-'+self.DR+'.fits.gz'
            self.catalog = self.survey+'_'+self.DR
        elif (self.DR in ['dr3', 'dr3_utah_ngc', 'dr5-eboss', 'dr5-eboss2', 'dr3_utah_sgc', 'eboss_combined']):
            inputdir     =  '/Volumes/TimeMachine/data/eboss/sysmaps/ccdfiles/'
            self.ccds    =  inputdir + 'survey-ccds-'+self.DR+'.fits'
            self.catalog = self.survey+'_'+self.DR
        elif (self.DR in ['90prime-new', 'decam-dr8', 'mosaic-dr8', 'dr8_combined']):
            inputdir     =  '/Volumes/TimeMachine/data/DR8/ccds/'
            self.ccds    =  inputdir + 'ccds-annotated-'+self.DR+'.fits'
            self.catalog = self.survey+'_'+self.DR            
        else:
            raise RuntimeError("Data Realease seems wrong") 

        # Bands inputs
        # MR fix the coefs, remove redundant vars
        # extc = {'r':2.165, 'z':1.211, 'g':3.214} # galactic extinction correction
        if band == 'g':
            self.extc = 3.214  #/2.751
        if band == 'r':
            self.extc = 2.165  #/2.751
        if band == 'z':
            self.extc = 1.211  #/2.751

# ------------------------------------------------------------------
# ------------ VALIDATION TESTS ------------------------------------
# ------------------------------------------------------------------
# Note: part of the name of the function should startw with number valXpX 
# modified by MR: only generate the maps
#
def generate_maps(sample):
    '''
       generate ivar, airmass, seeing, count and sky brightness map
       Aug 7,2019: call the simple version for dr8

    '''
    nside     = sample.nside       # Resolution of output maps
    nsideSTR  = str(nside)         # same as nside but in string format
    nsidesout = None               # if you want full sky degraded maps to be written
    ratiores  = 1                  # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode      = 1                  # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    pixoffset = 0                  # How many pixels are being removed on the edge of each CCD? 15 for DES.
    oversamp  = str(ratiores)      # ratiores in string format
    
    band           = sample.band
    catalogue_name = sample.catalog
    fname          = sample.ccds    
    localdir       = sample.localdir
    extc           = sample.extc

    #Read ccd file 
    print('working on ', fname)
    tbdata = pyfits.open(fname)[1].data
    # ------------------------------------------------------
    # Obtain indices that satisfy filter / photometric cuts
    #
    auxstr='band_'+band
    sample_names = [auxstr]
    if(sample.DR in ['DR3', 'DR5', 'eboss21', 'eboss22', 'eboss23', 'eboss25']):
        #inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
        inds = np.where((tbdata['filter'] == band)) #& (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
    elif(sample.DR == 'DR4'):
        inds = np.where((tbdata['filter'] == band) & (tbdata['photometric'] == True) & (tbdata['bitmask'] == 0)) 
    elif (sample.DR in ['DR5', 'DR7', 'dr3', 'dr3_utah_ngc', 'dr3_utah_sgc', 'dr5-eboss', 'dr5-eboss2', 'eboss_combined',\
                        '90prime-new', 'decam-dr8', 'mosaic-dr8', 'dr8_combined']):
        inds = np.where(tbdata['filter'] == band)
    else:
        sys.exit('something is wrong')
              
    #Read data 
    #obtain invnoisesq here, including extinction 
    nmag = Magtonanomaggies(tbdata['galdepth']-extc*tbdata['ebv'])/5.
    ivar= 1./nmag**2.
    
    # ref. MARC
    # compute CCDSKYMAG. since it's missing in DR 7
    # Aug 7, 2019: same issue in dr8
    ccdskymag = -2.5*np.log10(tbdata['ccdskycounts']/tbdata['pixscale_mean']/tbdata['pixscale_mean']/tbdata['exptime'])\
              + tbdata['ccdzpt']

    #
    # hits ~ count for fracgood
    hits = np.ones_like(ivar)

    # Fix fwhm unit depending on the camera
    pix2arcsec = {'decam':0.2637,
                  'mosaic':0.26,
                  '90prime':0.454}
    cameras = np.unique(tbdata['camera'])
    print(f'cameras in this ccd file {cameras}')
    for camera_i in cameras:
        if camera_i not in pix2arcsec.keys():
             raise ValueError(f'{camera_i} not available')
        else:
             print(f'fix fwhm unit for camera={camera_i}')
             mask_camera_i = tbdata['camera'] == camera_i
             tbdata['fwhm'][mask_camera_i] *= pix2arcsec[camera_i]
           
    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [ ('ivar',  '', 'total'),
                                ('hits',  '', 'total'),
                                ('hits',  '', 'fracdet'),
                                ('fwhm',  '', 'mean'),
                                #('fwhm', '', 'min'),
                                #('fwhm', '', 'max'),
                                ('airmass',   '', 'mean'),   # no airmass in DR 7, so COMMENT it out
                                ('exptime',   '', 'total'),
                                ('ccdskymag', '', 'mean'),
                                ('mjd_obs',   '', 'min')
                                #('mjd_obs', '', 'mean')
                                #('mjd_obs', '', 'max')
                              ]

 
    # What properties to keep when reading the images? 
    # Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate quicksip subroutines were implemented 
    #propertiesToKeep = [ 'filter', 'airmass', 'fwhm','mjd_obs','exptime','ccdskymag',\
    #                     'ra', 'dec', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1',\
    #                    'cd1_2', 'cd2_1', 'cd2_2','width','height']
    #propertiesToKeep = [ 'filter', 'fwhm','mjd_obs','exptime', 'airmass', #'ccdskymag',
    #                     'ra', 'dec', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1',
    #                    'cd1_2', 'cd2_1', 'cd2_2','width','height']
    propertiesToKeep = [ 'filter', 'fwhm','mjd_obs','exptime', 'airmass', #'ccdskymag',
                         'ra', 'dec', 'ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3']
 
    # Create big table with all relevant properties. 

    #tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar], names = propertiesToKeep + [ 'ivar'])
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar, ccdskymag, hits],
                                       names = propertiesToKeep + [ 'ivar', 'ccdskymag', 'hits'])
    #tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep], names = propertiesToKeep)
 
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values (use new version by MARCM for BASS and MzLS) 
    #project_and_write_maps(mode, propertiesandoperations, tbdata,
    #                       catalogue_name, localdir, sample_names, inds,
    #                       nside, ratiores, pixoffset, nsidesout)
    project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalogue_name, localdir, sample_names, inds, nside)

