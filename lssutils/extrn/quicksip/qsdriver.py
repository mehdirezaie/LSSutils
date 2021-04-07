"""
    The driver code that uses QuickSip to generate templates for 
    imaging attributes from CCD files
"""
import os
import numpy as np
import fitsio as ft
import pandas as pd

from ..galactic.hpmaps import logHI, SFD98, Gaia
from .quicksip import project_and_write_maps_simp


def IvarToDepth(ivar):
    """ change IVAR to DEPTH """
    depth = nanomaggiesToMag(5./np.sqrt(ivar))
    return depth

def nanomaggiesToMag(nm):
    """ nano maggies to magnitude """
    return -2.5 * (np.log10(nm) - 9.)


def Magtonanomaggies(mag):
    """ Change Mag to nanomaggies """
    return 10.0**(-mag/2.5+9.)

def fixdtype(data_in, dtype):
    """ fix dtype of a numpy structured array """
    m = data_in.size
    data_out = np.zeros(m, dtype=dtype)
    for name in dtype.names:
        data_out[name] = data_in[name].astype(dtype[name])
    return data_out  
    
def combine_ccds(ccds, output):
    """ Combines CCD annotated files
  
    """     
    columns = ['camera', 'filter', 'fwhm', 'mjd_obs', 'exptime', 
                'ra', 'dec', 'ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3',
                'galdepth', 'ebv', 'airmass', 'ccdskycounts', 'pixscale_mean', 'ccdzpt']
   
    dtype = np.dtype([('camera', '<U7'),('filter', '<U1'), ('exptime', '>f4'), ('mjd_obs', '>f8'), 
                      ('airmass', '>f4'), ('fwhm', '>f4'), ('ra', '>f8'), ('dec', '>f8'), ('ccdzpt', '>f4'),
                      ('ccdskycounts', '>f4'), ('ra0', '>f8'), ('dec0', '>f8'), ('ra1', '>f8'),
                      ('dec1', '>f8'), ('ra2', '>f8'), ('dec2', '>f8'), ('ra3', '>f8'), ('dec3', '>f8'),
                      ('pixscale_mean', '>f4'), ('ebv', '>f4'), ('galdepth', '>f4')])
   
    # read each ccd file > fix its dtype > move on to the next
    ccds_data = []
    for ccd_i in ccds:
        
        print('working on .... %s'%ccd_i.split('/')[-1])
        data_in = ft.FITS(ccd_i)[1].read(columns=columns)

        data_out = fixdtype(data_in, dtype)
        
        in_diff = np.setdiff1d(dtype.descr, data_in.dtype.descr)
        out_diff = np.setdiff1d(dtype.descr, data_out.dtype.descr)
        print(f'number of ccds in this file: {data_in.size}')
        print(f'different dtypes (before): {in_diff}')
        print(f'different dtypes (after): {out_diff}')
        ccds_data.append(data_out)    
        
    ccds_data_c = np.concatenate(ccds_data)
    print(f'Total number of combined ccds : {ccds_data_c.size}')
    
    ft.write(output, ccds_data_c, clobber=True)
    print(f'wrote the combined ccd file: {output}')
    
        
def make_maps(ccdfile, nside, bands, catalog, outputdir):
    """ Make imaging maps for each band
    """
    mode = 1                  # 1: fully sequential, 2: parallel then sequential, 3: fully parallel


    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesToKeep = [ 'filter', 'fwhm', 'mjd_obs', 'exptime', 'airmass', # ccdskymag
                         'ra', 'dec', 'ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3']
    
    propertiesandoperations = [ ('nobs', '', 'total'),
                                ('ivar',  '', 'total'),
                                ('nobs',  '', 'fracdet'),
                                ('fwhm',  '', 'mean'),
                                ('fwhm', '', 'min'),
                                ('fwhm', '', 'max'),
                                ('airmass',   '', 'mean'),
                                ('exptime',   '', 'total'),
                                ('ccdskymag', '', 'mean'),
                                ('mjd_obs',   '', 'min'),
                                ('mjd_obs', '', 'mean'),
                                ('mjd_obs', '', 'max')
                              ]

   
    tbdata = ft.read(ccdfile)
    columns = tbdata.dtype.names    
    nobs = np.ones(tbdata.size) # nobs is missing
    
    # ref. MARC
    # compute CCDSKYMAG. since it's missing in DR 7
    # Aug 7, 2019: same issue in dr8
    ccdskymag = -2.5*np.log10(tbdata['ccdskycounts']/tbdata['pixscale_mean']/tbdata['pixscale_mean']/tbdata['exptime'])\
              + tbdata['ccdzpt']

    # fwhm unit
    pix2arcsec = {'decam':0.2637, 'mosaic':0.26, '90prime':0.454}
    cameras = np.unique(tbdata['camera'])
    print(f'cameras in this ccd file {cameras}')
    for camera_i in cameras:
        if camera_i.strip() not in pix2arcsec.keys():
             raise ValueError(f'{camera_i} not available')
        else:
             print(f'fix fwhm unit for camera={camera_i}')
             mask_camera_i = tbdata['camera'] == camera_i
             tbdata['fwhm'][mask_camera_i] *= pix2arcsec[camera_i.strip()]

    # Obtain indices that satisfy filter / photometric cuts 
    sample_names = []
    inds = []
    
    for band in bands:        
        
        good = tbdata['filter'] == band
        if 'photometric' in columns:
            good &= tbdata['photometric'] == True
        if 'bitmaks' in columns:
            good &= tbdata['bitmask'] == 0     
            
        if good.sum() > 0:
            inds.append(np.argwhere(good).flatten())
            sample_names.append('band_%s'%band)
        else:
            print(f'there is no {band} in the ccd file')

    extc = {'r':2.165, 'z':1.211, 'g':3.214}

    # obtain ivar including extinction
    nmag = np.empty(tbdata.size)
    nmag[:] = np.nan
    for i, band in enumerate(bands):
        print(i, band, inds[i][:5], len(inds[i]))
        nmag[inds[i]] = Magtonanomaggies(tbdata['galdepth'][inds[i]]-extc[band]*tbdata['ebv'][inds[i]])/5.0
    ivar = 1./(nmag*nmag)


    # Create big table with all relevant properties. 
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar, ccdskymag, nobs],
                                       names = propertiesToKeep + [ 'ivar', 'ccdskymag', 'nobs'])
 
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    project_and_write_maps_simp(mode, propertiesandoperations, tbdata, catalog, 
                                outputdir, sample_names, inds, nside) 
    
    
def get_sysname(filename):
    """ infer systematic name, opertation and band
    """
    splits = filename.split('/')[-1].split('_')
    band = splits[2]    
    sysname = splits[5]    
    operation = splits[-1].split('.')[0]

    return '_'.join([sysname, operation, band])

def make_hp(nside, hpix, signal):
    """ make a healpix map given hpix and signal
    """
    hpmap = np.empty(12*nside*nside)
    hpmap[:] = np.nan    
    hpmap[hpix] = signal
    
    return hpmap


def combine_fits(input_maps, nside, add_galactic=False, write_to=None):
    """ Combine the imaging maps (.fits) into a hdf5 file
    """
    df = {}

    for map_ in input_maps:
        sysname = get_sysname(map_)

        dmap_ = ft.read(map_)
        hpmap_ = make_hp(nside, dmap_['PIXEL'], dmap_['SIGNAL'])

        if 'ivar' in sysname:
            hpmap_ = IvarToDepth(hpmap_)
            sysname = sysname.replace('ivar', 'depth')
            print(f'changed {sysname}')
            
        df[sysname] = hpmap_
        print('.', end='')
        
    if add_galactic:
        # add galactic foregrounds
        gaia = Gaia(nside_out=nside)
        df['nstar'] = gaia.map

        sfd = SFD98(nside_out=nside)
        df['ebv'] = sfd.map

        loghi = logHI(nside_out=nside)
        df['loghi'] = loghi.map    
        
    df = pd.DataFrame(df)
    
    if write_to is not None:

        dirname = os.path.dirname(write_to)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        df.to_hdf(write_to, key="templates")
        
    return df
