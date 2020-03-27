'''
    Data Visualization tools

    > to reset the defaults
    
    from cycler import cycler
    default_cycler = (cycler(color=['purple', 'royalblue', 'lime', 'darkorange', 'crimson']) +
                      cycler(linestyle=['-', '--', ':', '-.', '-']))
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=default_cycler)
    

'''
#import matplotlib
#matplotlib.use('Agg')


import matplotlib.pyplot as plt
import numpy  as np
import healpy as hp
import seaborn as sns


from matplotlib.gridspec import GridSpec
from matplotlib.projections.geo import GeoAxes
from matplotlib import cm
from matplotlib.colors import ListedColormap

from .utils import binit, binit_jac, radec2hpix



def plot_corrmax(corrmatrix, title, xlabels, pdfname):
    '''
    columns = lab.catalogs.datarelease.cols_dr8_ccd
    xlabels = lab.catalogs.datarelease.fixlabels(columns, addunit=False)

    df = pd.read_hdf('/home/mehdi/data/templates/dr8pixweight-0.32.0_combined256.h5')[columns]
    df['ngal/nran'] = 0
    dfnumpy = df.to_numpy()

    kw = {'verbose':False}
    ngal = hp.read_map('/home/mehdi/data/formehdi/dr8_elgsv_ngal_pix_0.32.0-colorbox.hp.256.fits', **kw)
    frac = hp.read_map('/home/mehdi/data/formehdi/dr8_frac_pix_0.32.0-colorbox.hp.256.fits', **kw)

    masks = {}
    masks['bmzls'] = hp.read_map('/home/mehdi/data/formehdi/dr8_mask_eboss_bmzls_pix_0.32.0-colorbox.hp.256.fits', **kw) > 0
    masks['decals'] = hp.read_map('/home/mehdi/data/formehdi/dr8_mask_eboss_decals_pix_0.32.0-colorbox.hp.256.fits', **kw) > 0 

    nnbar = {}
    nnbar['nnbar_bmzls'] = lab.utils.makedelta(ngal, frac, masks['bmzls']) + 1
    nnbar['nnbar_decals'] = lab.utils.makedelta(ngal, frac, masks['decals']) + 1

    corrs = {}
    for region in ['bmzls', 'decals']:
        df_region = dfnumpy.copy()
        df_region[:,-1] = nnbar[f'nnbar_{region}']    

        corrs[region] = lab.utils.corrmatrix(df_region[masks[region], :], 
                                             estimator='pearsonr')    
        del df_region


    cbar_label = {'bmzls':'BASS/MzLS',
                 'decals':'DECaLS'}
    for region in corrs:
        corr_reg = corrs[region]

        pdfname = f'pcorr_{region}.pdf'    
        lab.dataviz.plot_corrmax(corr_reg, cbar_label[region], xlabels, pdfname)
    '''
    params = {
    'axes.spines.right':False,
    'axes.spines.top':False,
    'axes.labelsize': 12,
    #'text.fontsize': 8,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': True,
    #'figure.figsize': [6, 4],
    'font.family':'serif'
    }
    plt.rcParams.update(params)
    
    #--- setup figure
    fig, ax = plt.subplots(figsize=(6, 4))
    mask = np.zeros_like(corrmatrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    #--- colorbar 
    kw = dict(mask=mask, cmap=plt.cm.seismic, 
              center=0, vmin=-0.5, vmax=0.5,
              square=True, linewidths=.5, 
              cbar_kws={"shrink": .5, 
                        "ticks":[-1, -0.5, 0, 0.5, 1], 
                        #"label":'PCC',
                        "orientation":'vertical',
                        'extend':'both'},
              xticklabels=xlabels,
              yticklabels=xlabels+['ngal/nran'],
              ax=ax)    
    sns.heatmap(corrmatrix, **kw)
    
    ax.set_ylim(ymax=1) # Hack to remove the EBV from y axis
    ax.text(0.55, 0.95, title,
            color='k', transform=ax.transAxes,
            fontsize=15) 
    fig.savefig(pdfname, bbox_inches='tight')


def plot_grid(galm, extent=[100, 275, 10, 70], cmap=plt.cm.Blues, 
              interpolation='none' , aspect='auto', **kwarg):    
    nside   = hp.get_nside(galm)
    ra, dec = np.meshgrid(np.linspace(extent[0], extent[1], 500),
                          np.linspace(extent[2], extent[3], 500))
    hpix    = radec2hpix(nside, ra, dec)
    mymap   = galm[hpix]
    mymap   = mymap[::-1, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    map1    = ax.imshow(mymap, extent=extent, cmap=cmap, 
                        interpolation=interpolation, aspect=aspect, 
                        **kwarg)
    fig.colorbar(map1, label='Nqso', extend='both')
    
    
def hpmollview(map1, unit, ax, smooth=False, cmap=plt.cm.bwr, **kw):
    '''
    Example:
    
    kw  = dict(min=-0.5, max=.5, cmap=dataviz.mycolor(), rot=-85, title='')
    fig, ax = plt.subplots(nrows=2, figsize=(7, 7))
    plt.subplots_adjust(hspace=0.05)
    dataviz.hpmollview(d0, r'$\delta_{\rm ELG}$', ax[0], **kw)
    dataviz.hpmollview(d1, r'$\delta_{\rm LRG}$', ax[1], **kw)
    plt.savefig('delta_dr8.png', bbox_inches='tight', dpi=300)
    
    '''
    cmap.set_over(cmap(1.0))
    cmap.set_under('w')
    cmap.set_bad('white')
    # galactic plane
    r = hp.Rotator(coord=['G','C'])
    theta_gal, phi_gal = np.zeros(1000)+np.pi/2, np.linspace(0, 360, 1000)
    theta_cl,  phi_cl  = r(theta_gal, phi_gal)
    
    plt.sca(ax)
    if smooth:map1 = hp.smoothing(map1, fwhm=np.deg2rad(0.5))
    hp.mollview(map1, hold=True, unit=unit, cmap=cmap, **kw)
    hp.projplot(theta_cl, phi_cl, 'r.', alpha=1.0, markersize=1.)
    hp.graticule(dpar=45, dmer=45, coord='C', verbose=False)    


def mycolor():
    # colors
    top       = cm.get_cmap('Blues_r', 128)
    bottom    = cm.get_cmap('Oranges', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcolors[127,:] = [1.0, 1.0, 1., 1.0]
    newcolors[128,:] = [1.0, 1.0, 1., 1.0]
    newcolors[129,:] = [1.0, 1.0, 1., 1.0]
    newcolors[130,:] = [1.0, 1.0, 1., 1.0]
    return ListedColormap(newcolors, name='OrangeBlue')
    

def cm2inch(cm):
    """Centimeters to inches"""
    return cm *0.393701


def mollview(m, vmin, vmax, unit, use_mask=False, 
             maskname=None, rotate=2/3*np.pi, xsize=1000,
             width=7, figax=None, colorbar=False, cmap=plt.cm.bwr,
             galaxy=False, extend='both',**kwargs):
    '''
        (c)Andrea Zonca, https://github.com/zonca/paperplots 
        modified by Mehdi Rezaie for galaxy counts
        
        Matplotlib has higher freedom than healpy.mollview
        
        one could use rasterized=True to improve the quality
        for colorbar options, see
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.colorbar.html

        Example:
        
        unit = r'$\delta$'
        vmin = -0.5
        vmax = 0.5

        fig = plt.figure(figsize=(6, 7))
        # matplotlib is doing the mollveide projection
        ax  = fig.add_subplot(211, projection='mollweide')
        ax1 = fig.add_subplot(212, projection='mollweide')
        spacing = 0.01
        plt.subplots_adjust(bottom=spacing, top=1-spacing, 
                            left=spacing, right=1-spacing,
                            hspace=0.0)


        dataviz.mollview(d0, vmin, vmax, unit, figax=[fig, ax], cmap=dataviz.mycolor())
        dataviz.mollview(d1, vmin, vmax, unit, figax=[fig, ax1], cmap=dataviz.mycolor(), colorbar=True)
        plt.savefig('./delta_dr8.png', bbox_inches='tight', dpi=300)
        
        
        
        data = ft.read('/Users/mehdi/Dropbox/forMehdi/pixweight_ar-dr8-0.32.0-elgsv.fits')
        wsys = hp.read_map('/Users/mehdi/Downloads/nn-weights.hp256.fits')
        ebv = hp.reorder(data['EBV'], n2r=True)
        ebv[ebv==-1] = np.nan

        depth = hp.reorder(data['GALDEPTH_G'], n2r=True)
        depth[depth==-1] = np.nan

        frac = hp.reorder(data['FRACAREA'], n2r=True)
        elg = hp.reorder(data['SV'], n2r=True)
        elg[frac==0.0] = np.nan
        fig = plt.figure(figsize=(5, 7))
        # matplotlib is doing the mollveide projection
        ax0  = fig.add_axes([0, 0, 1., 1],       projection='mollweide')
        ax1  = fig.add_axes([0., -0.3, 1., 1], projection='mollweide')
        ax2  = fig.add_axes([0., -0.6, 1., 1], projection='mollweide')
        ax3  = fig.add_axes([0., -.9, 1., 1], projection='mollweide')
        kw = {'unit':'', 'galaxy':False}
        dataviz.mollview(elg, vmin=3000, vmax=7500, 
                         cmap=dataviz.mycolor(), 
                         figax=[fig, ax0], **kw)

        dataviz.mollview(depth, vmin=0, vmax=2000, 
                         cmap=plt.cm.viridis, 
                         figax=[fig, ax1], **kw)

        dataviz.mollview(ebv, vmin=0.0, vmax=0.05, 
                         cmap=plt.cm.viridis,
                         figax=[fig, ax2], **kw)

        dataviz.mollview(wsys, vmin=0.8, vmax=1.2, 
                         cmap=dataviz.mycolor(),
                         figax=[fig, ax3], **kw)

        for axi in [ax0, ax1, ax2, ax3]:axi.set(xticks=[], yticks=[])
        
    '''    
    nside     = hp.npix2nside(len(m))
    rotatedeg = np.degrees(rotate)    
    ysize     = xsize/2.                       # ratio is always 1/2

    
    # galactic plane
    if galaxy:        
        r = hp.Rotator(coord=['G','C'])
        theta_gal, phi_gal = np.zeros(1000)+np.pi/2, np.linspace(0, 360, 1000)
        theta_cl,  phi_cl  = r(theta_gal, phi_gal)
    
    # set up the grid
    theta      = np.linspace(np.pi,  0.0,   ysize)
    phi        = np.linspace(-np.pi, np.pi, xsize)
    longitude  = np.radians(np.linspace(-180, 180, xsize))
    latitude   = np.radians(np.linspace(-90, 90, ysize))

    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix   = hp.ang2pix(nside, THETA, PHI)

    if use_mask:
        m.mask    = np.logical_not(hp.read_map(maskname))
        grid_mask = m.mask[grid_pix]
        grid_map  = np.ma.MaskedArray(m[grid_pix], grid_mask)
    else:
        grid_map = m[grid_pix]

    # class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
    #     """Shifts labelling by pi
    #     Shifts labelling from -180,180 to 0-360"""
    #     def __call__(self, x, pos=None):
    #         if x != 0:
    #             x *= -1
    #         if x < 0:
    #             x += 2*np.pi
    #         return GeoAxes.ThetaFormatter.__call__(self, x, pos)

    class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
        """
        Shifts the zero labelling by 2/3 pi to left
        """
        def __call__(self, x, pos=None):
            x += 2*np.pi/3
            return GeoAxes.ThetaFormatter.__call__(self, x, pos)    

    def rot(grid_map, angle=60):
        
        n = int(angle*(grid_map.shape[1]/360))
        m = grid_map.shape[1]-n        

        grid_new = np.zeros_like(grid_map)
        grid_new[:, :m] = grid_map[:, n:]
        grid_new[:, m:] = grid_map[:, :n]
        return grid_new



    if figax is None:
        fig = plt.figure(figsize=(width, 2/3*width))
        # matplotlib is doing the mollveide projection
        ax = fig.add_subplot(111, projection='mollweide')
        spacing = 0.01
        plt.subplots_adjust(bottom=spacing, top=1-spacing, 
                            left=spacing,   right=1-spacing,
                            hspace=0.0)
    else:
        fig = figax[0]
        ax  = figax[1]

    # rasterized makes the map bitmap while the labels remain vectorial
    # flip longitude to the astro convention
    image = ax.pcolormesh(longitude, latitude, rot(grid_map, rotatedeg), 
                           vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)
    
    if galaxy:
        ax.scatter(phi_cl-2/3*np.pi, np.pi/2-theta_cl, 2, color='r')
        ax.scatter(phi_cl+4/3*np.pi, np.pi/2-theta_cl, 2, color='r')
    # graticule
    ax.set_longitude_grid(60)
    ax.set_latitude_grid(30)
    ax.xaxis.set_major_formatter(ThetaFormatterShiftPi())

    # colorbar
    if colorbar:
        #cax = plt.axes([.9, 0.2, 0.02, 0.6])  # vertical
        cax = plt.axes([0.2, 0.0, 0.6, 0.02])  # horizontal
        cb  = fig.colorbar(image, cax=cax, label=unit, fraction=0.15,
                           shrink=0.6, pad=0.05, ticks=[vmin, vmax], # 0.5*(vmax+vmin), 
                           orientation='horizontal', extend=extend)        
        #cb = fig.colorbar(image, orientation='horizontal', shrink=.6, pad=0.05, ticks=[vmin, vmax])
        #cb.ax.xaxis.set_label_text(unit)
        cb.ax.xaxis.labelpad = -8
        # workaround for issue with viewers, see colorbar docstring
        cb.solids.set_edgecolor("face")

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # remove tick labels
    #ax.xaxis.set_ticklabels([])
    #ax.yaxis.set_ticklabels([])

    # remove grid    
    #ax.xaxis.set_ticks([])
    #ax.yaxis.set_ticks([])

    # remove white space around figure
    #spacing = 0.01
    #plt.subplots_adjust(bottom=spacing, top=1-spacing, 
    #                    left=spacing, right=1-spacing,
    #                    hspace=0.0)
    ax.grid(True)
    #plt.show()




def get_selected_maps(files1=None, tl=['eBOSS QSO V6'], 
                      verbose=False, labels=None, ax=None,
                     hold=False, saveto=None):
    '''
    from LSSutils.catalogs.datarelease import cols_dr8 as labels
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(18, 12), sharey=True)
    ax = ax.flatten()

    i = 0
    for cap in [ 'elg', 'lrg']: # ngc.all
        for key in ['decals','decaln', 'bmzls']:
            mycap = cap+'_'+key+'_'+'256' # NGC_0.8
            get_selected_maps(glob(f'/home/mehdi/data/alternative/results_{cap}/ablation_{key}/dr8.log_fold*.npy'),
                              ['DR8 '+mycap], labels=labels, ax=ax[i], hold=True)
            i += 1
    #plt.savefig('./maps_selected_eboss.pdf', bbox_inches='tight')
    plt.show()   
    
    '''
    def get_all(ablationlog):
        d = np.load(ablationlog, allow_pickle=True).item()
        indices = None
        for il, l in enumerate(d['validmin']):
            m = (np.array(l) - d['MSEall']) > 0.0
            #print(np.any(m), np.all(m))
            if np.all(m):
                #print(il, d['indices'][il])
                #print(il, [lbs[m] for m in d['indices'][il]])
                #break
                indices = d['indices'][il]
                break
            if (il == len(d['validmin'])-1) & (np.any(m)):
                indices = [d['indices'][il][-1]]       
        # return either None or indices
        num_f   = len(d['importance'])+1
        FEAT    = d['importance'] + [i for i in range(num_f)\
              if i not in d['importance']]
        if indices is not None:
            return FEAT[num_f-len(indices):], num_f
        else:
            return FEAT[num_f:], num_f
        
    def add_plot(axes, ax, **kw):
        '''
        Aug 20: https://stackoverflow.com/questions/52876985/
                matplotlib-warning-using-pandas-dataframe-plot-scatter/52879804#52879804
        '''
        
        m = 0
        for i in range(len(axes)):
            if axes[i] is np.nan:
                continue
            else:
                #print(axes[i])                
                n = len(axes[i])
                colors = np.array([plt.cm.Reds(i/n) for i in range(n)])
                m += n
                for j in range(n):
                    ax.scatter(i, axes[i][j], c='k', marker='x')                    
                    ax.scatter(i, axes[i][j], c=[colors[j]], marker='o', **kw)   
   
                    
    def get_axes(files, verbose=False):    
        axes = []
        for filei in files:
            axi, num_f = get_all(filei)
            if verbose:print(axi)
            if axi is not None:
                axes.append(axi)
            else:
                axes.append(np.nan)
        return axes, num_f
    
    if verbose:print(files1)
    axes1, num_f = get_axes(files1, verbose=verbose)
     
        
    if ax is None:
        fig, ax = plt.subplots(ncols=1, 
                               sharey=True, 
                               figsize=(6, 4))
    ax = [ax]
    add_plot(axes1, ax[0])
    ax[0].set_yticks(np.arange(num_f))
    ax[0].set_yticklabels(labels)
    ax[0].set_xticks(np.arange(5))
    ax[0].set_xticklabels(['1', '2', '3', '4', '5'])

    for i,axi in enumerate(ax):
        #axi.set_title(tl[i])
        axi.grid()
        axi.set_xlabel(tl[i]+' Partition-ID')    
    if saveto is not None:plt.savefig(saveto, bbox_inches='tight')
    if not hold:plt.show()



def read_ablation_file(filename, 
                       allow_pickle=True):
    ab1     = np.load(filename, allow_pickle=allow_pickle).item()
    INDICES = ab1['indices']
    VALUES  = ab1['validmin']
    num_f   = len(ab1['importance'])+1
    FEAT    = ab1['importance'] + [i for i in range(num_f)\
                  if i not in ab1['importance']]
    
    matric_dict = {}
    for i in range(len(INDICES)):
        for j in range(len(VALUES[i])):
            matric_dict[str(i)+'-'+str(INDICES[i][j])] = VALUES[i][j]

    matric = np.zeros(shape=(num_f, num_f))
    for i in range(num_f-1):
        for j, sys_i in enumerate(FEAT):
            if str(i)+'-'+str(sys_i) in matric_dict.keys():
                matric[i,j] = (matric_dict['%d-%d'%(i,sys_i)][0]/ab1['MSEall'])-1.

    return matric, FEAT

class plot_multivariate_params(object):
    """"
    files = glob('/home/mehdi/data/eboss/v6/imag_splits/regression/mult_ngc.*/regression_log.npy')
    labels = {'0':'i<=%.2f'%19.948435,
             '1':'%.2f<i<=%.2f'%(19.948435, 20.585423),
             '2':'%.2f<i<=%.2f'%(20.585423, 21.108092),
             '3':'%.2f<i'%21.108092}
    colnames = ['b'] + colnames

    m  = ['.', 'x', 's', '^']
    ls = ['-', '--', ':', '-.']
    mult_params = dataviz.plot_multivariate_params(colnames)

    for i,f_i in enumerate(files):

        d_i      = np.load(f_i, allow_pickle=True).item()    
        lparams  = d_i['params']['lin'] # 
        nparams  = np.arange(len(lparams[0]))+(-0.1+0.05*i)
        arrays   = (nparams, lparams[0], np.diag(lparams[1])**0.5)
        kwargs   = dict(marker=m[i], ls=ls[i], label=labels[str(i)])

        mult_params(*arrays, **kwargs)

    mult_params.show()
    
    """
    def __init__(self, colnames, ylim1=(0.95, 1.05), ylim2=(-0.4, 0.4), d=0.01):
        #
        fig = plt.figure(figsize=(10, 6))
        gs  = GridSpec(2, 1, height_ratios=[1, 4], figure=fig)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xticks([i for i in range(len(colnames))])
        ax2.set_xticklabels(colnames, rotation=90)
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d/2, +d/2), **kwargs)        # top-left diagonal
        kwargs.update(transform=ax2.transAxes)            # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d/2, 1 + d/2), **kwargs)  # bottom-left diagonal
        ax2.grid(True)
        ax1.set_ylim(*ylim1)
        ax2.set_ylim(*ylim2)
        self.ax1 = ax1
        self.ax2 = ax2
        
    def __call__(self, *arrays, **kwargs):        
        self.ax1.errorbar(*arrays, **kwargs)
        self.ax2.errorbar(*arrays, **kwargs)
    
    def show(self):
        self.ax2.legend(bbox_to_anchor=(0.8, 1.2))
        plt.show()
    

def ablation_plot(filename,  
                  labels=None, 
                  allow_pickle=True,
                  annot=True,
                  hold=False,
                  saveto=None,
                  ax=None):    
    
    matric, FEAT = read_ablation_file(filename, allow_pickle=allow_pickle)
    
    matric *= 1.e4
    if labels is None:
        labels = [str(i) for i in range(len(FEAT))]
        
    xlabels = [labels[j] for j in FEAT]
    mask = ~np.zeros_like(matric, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = False
    mask[matric==0.0] = False
    vmin = np.minimum(np.abs(np.min(matric)), np.abs(np.max(matric))) #* 0.1
    print(np.max(matric), np.min(matric))
    # Set up the matplotlib figure
    if ax is None:
        f, ax = plt.subplots()
        
    #plt.title('Correlation Matrix of DR5')
    # Generate a custom diverging colormap
    kw = dict(mask=~mask, cmap=plt.cm.seismic, xticklabels=xlabels, #PRGn_r,get_cmap('PRGn_r', 20)
               yticklabels=xlabels[::-1], 
               vmax=20, center=0.0,#vmin=-1.*vmin, vmax=vmin, center=0.0,
               square=True, linewidths=.5, 
               cbar_kws={"shrink": .5, 
               "label":r'$10^{4} \times \delta$MSE', "extend":"max"},
               ax=ax)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matric, **kw)
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticks([])
    ax.xaxis.tick_top()
    
    # add annotation
    if annot:        
        bbox_props = dict(boxstyle="rarrow", fc="white", ec="k", lw=2)
        t = ax.text(0.4, 0.2, "Importance",
                    ha="center", va="center", rotation=0,
                    transform=ax.transAxes,
                    bbox=bbox_props, fontsize=15)
        bb = t.get_bbox_patch()
        bb.set_boxstyle("rarrow", pad=0.6)
        t1 = ax.text(0.1, 0.5, "Iteration",
                    ha="center", va="center", rotation=-90,
                    transform=ax.transAxes,
                    bbox=bbox_props, fontsize=15)
        bb1 = t1.get_bbox_patch()
        bb1.set_boxstyle("rarrow", pad=0.6)
        
    if saveto is not None: plt.savefig(saveto, bbox_inches='tight') 
    if not hold:plt.show()

def ablation_plot_all(files, labels=None, title=None, saveto=None, hold=False):    
    '''
    
    from LSSutils.catalogs.datarelease import cols_dr8 as labels
    i = 0
    for cap in [ 'elg', 'lrg']: # ngc.all
        for key in ['decaln', 'decals', 'bmzls']:
            mycap = cap+'_'+key+'_'+'256' # NGC_0.8
            ablation_plot_all(glob(f'/home/mehdi/data/alternative/results_{cap}/ablation_{key}/dr8.log_fold*.npy'),
                              title='DR8 '+mycap, labels=labels)
            i += 1
    #plt.savefig('./maps_selected_eboss.pdf', bbox_inches='tight')
    plt.show()   
    
    '''
    f = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 6, figure=f)
    gs.update(wspace=0.15, hspace=0.35)
    ax1 = plt.subplot(gs[0, 1:3])
    ax2 = plt.subplot(gs[0, 3:5])
    ax3 = plt.subplot(gs[1, 0:2])
    ax4 = plt.subplot(gs[1, 2:4])
    ax5 = plt.subplot(gs[1, 4:6])
    for ax in [ax2, ax4, ax5]:
        ax.tick_params(labelleft=False)
    ax = [ax1, ax2, ax3, ax4, ax5]    
    for i, axi in enumerate(ax):
        annot = True if i == 2 else False # make the annotation
        ablation_plot(files[i],  
                  labels=labels, 
                  allow_pickle=True,
                  annot=annot,
                  hold=True,
                  saveto=None,
                  ax=axi)

    ax1.text(0., 0.2, title, color='k', transform=ax1.transAxes)    
    # bbox_props = dict(boxstyle="rarrow", fc="white", ec="k", lw=2)
    # t = ax3.text(0.5, 0.1, "Importance",
    #             ha="center", va="center", rotation=0,
    #             transform=ax3.transAxes,
    #             bbox=bbox_props, fontsize=10)
    # bb = t.get_bbox_patch()
    # bb.set_boxstyle("rarrow", pad=0.6)
    # t1 = ax3.text(0.1, 0.5, "Iteration",
    #             ha="center", va="center", rotation=-90,
    #             transform=ax3.transAxes,
    #             bbox=bbox_props, fontsize=10)
    # bb1 = t1.get_bbox_patch()
    # bb1.set_boxstyle("rarrow", pad=0.6)    
    if saveto is not None:plt.savefig(saveto, bbox_inches='tight')
    if not hold:plt.show()
        
        
        
def plot_cell(filen, clsysname, labels, bins=None, 
              error=True, corrcoef=True, annot=True, 
              title=r'Cross C$_\ell$', saveto=None):
    '''
    Example:
    
    cap    = 'ngc'
    CAP    = cap.upper()
    filen  = lambda l:'/home/mehdi/data/eboss/v6/results_'+cap+'.all/clustering/cl_'+CAP+'_'+l+'.npy'
    clsys  = '/home/mehdi/data/eboss/v6/results_'+cap+'.all/clustering/cl_sys.npy'

    labels = dict(lb = ['v6_wosys', 'v6', 'v6_wosys_z0.8', 'v6_wosys_z1.1', 'v6_wosys_z1.3',
                        'v6_wosys_z1.5', 'v6_wosys_z1.7', 'v6_wosys_z1.9'],
                  lt = ['No correction', 'systot (0.8<z<2.2)', '0.8<z<1.1', 
                        '1.1<z<1.3', '1.3<z<1.5', '1.5<z<1.7', '1.7<z<1.9','1.9<z<2.2'],
                  c  = ['k', 'k', 'purple', 'royalblue', 'crimson', 'olive', 'g', 'darkorange'],
                  ls = 3*['-', '--', '-', '-.'],
                  mk = 3*['.', 'o', '^', 'x'])

    dataviz.plot_cell(filen, clsys, labels, corrcoef=False, title=CAP+'No correction')
    dataviz.plot_cell(filen, clsys, labels, corrcoef=True,  title=CAP+'No correction')
    
    '''

    clsys = np.load(clsysname, allow_pickle=True).item()
    if bins is None:bins = np.logspace(0, 2.71, 9)
    #plt.rc('font', size=20)
    fig, ax = plt.subplots(ncols=4, nrows=5, figsize=(20, 20), sharey=True, sharex=True)
    plt.subplots_adjust(hspace=0., wspace=0)
    ax = ax.flatten()
    
    lb = labels['lb']
    c  = labels['c']
    mk = labels['mk']
    ls = labels['ls']
    lt = labels['lt']
    n  = len(lb)-1
    for j, lbi in enumerate(lb):
        d = np.load(filen(lbi), allow_pickle=True).item()
        
        # show error-bar
        if error:
            if j==0:
                lb, clbe = binit_jac(d['clerr']['cljks'], bins=bins)
                for i in range(d['cross'].shape[0]):
                    ax[i].fill_between(lb, 1.e-13, clbe, color='grey', label=r'Error on C$^{g,g}$', alpha=0.2)

        # show cross power
        for i in range(d['cross'].shape[0]):
            l = np.arange(d['cross'][i, :].size)
            if corrcoef:
                cl= np.abs(d['cross'][i, :]) / np.sqrt((clsys['cross'][i, :]*d['auto']))
            else:
                cl= d['cross'][i, :]**2 / (clsys['cross'][i, :])
                
            lb, clb = binit(l, cl, bins=bins)
            #print(lb, clb)
            #a[i].plot(l, cl, color=color, label=label, linestyle=ls)
            ax[i].plot(lb, clb, color=c[j], label=lt[j], marker=mk[j], linestyle=ls[j])
            

    #
    if annot:
        ax[0].legend(**dict(ncol=5, frameon=False,
                     bbox_to_anchor=(0, 1.1, 4, 0.4), loc="lower left",
                    mode="expand", borderaxespad=0, fontsize=20, title=title))
        for i in range(d['cross'].shape[0]):
            ax[i].set_xscale('log')
            #ax[0].set_xlim(1, 3)
            if corrcoef:
                ax[i].set_ylim(-0.1, 1.)
            else:
                ax[i].set_ylim(8.e-9, 1.e-3)
                #ax[i].set_ylim(1.e-5, 1.e1)
                ax[i].set_yscale('log')
            ax[i].text(0.5, 0.8, 'X %s'%d['clabels'][i], transform=ax[i].transAxes)
            ax[i].grid()
            
        if corrcoef:
            ax[8].set_ylabel(r'$|C^{g,s}_{\ell}|/[C^{s,s}_{\ell} C^{g,g}_{\ell}]^{1/2}$')
        else:
            ax[8].set_ylabel(r'$(C^{g,s}_{\ell})^{2}/C^{s,s}_{\ell}$')
        ax[17].set_xlabel(r'$\ell$')                    
    if saveto is not None:plt.savefig(saveto, bbox_inches='tight')
        
def plot_cross_xi(config):
    '''
        from LSSutils.catalogs.datarelease import cols_eboss_v6_qso_simp as xticks
        config = {
            'crossxi':{
                 'files_names':['/home/mehdi/data/eboss/v6/results_ngc.all/clustering/xi_NGC_v6_z0.8.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/xi_NGC_v6_z1.1.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/xi_NGC_v6_z1.3.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/xi_NGC_v6_z1.5.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/xi_NGC_v6_z1.7.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/xi_NGC_v6_z1.9.npy'
                              ],
                 'xisys':'/home/mehdi/data/eboss/v6/results_ngc.all/clustering/xi_sys.npy',
                 'title':'NGC w Systot',
                 'labels':['0.8<z<1.1', '1.1<z<1.3','1.3<z<1.5', '1.5<z<1.7', '1.7<z<1.9', '1.9<z<2.2'],
                 'saveto':None,
                 'colors':None,
                 'xticks':xticks}          
             }

        dataviz.plot_cross_xi(config)
    
    '''
    # mpl.rcParams.update(mpl.rcParamsDefault)
    # params = {
    # 'axes.spines.right':True,
    # 'axes.spines.top':True,
    # 'axes.labelsize': 20,
    # #'text.fontsize': 8,
    # 'legend.fontsize': 15,
    # 'xtick.labelsize': 12,
    # 'ytick.labelsize': 12,
    # 'text.usetex': True,
    # #'figure.figsize': [4, 3],
    # 'font.family':'serif',
    # 'font.size':12
    # }
    # plt.rcParams.update(params)
    # plt.rc('xtick', labelsize='medium')
    # plt.rc('ytick', labelsize='medium')
    files_names = config['crossxi']['files_names']
    labels      = config['crossxi']['labels']
    num_files   = len(files_names)
    colors      = config['crossxi']['colors']
    if colors is None:
        colors = [plt.cm.Blues((i+2)/(num_files+1)) for i in range(num_files)]
        
    title      = config['crossxi']['title']    
    linestyles = 50*['-','-.','--', '-']
    markers    = 50*['^', 's', 'd', 'o']
    file_xisys = config['crossxi']['xisys']
    factor     = 1.e3
    
    nrows  = len(config['crossxi']['xticks']) // 3
    pltarg = dict(ncols=3, nrows=nrows, sharex=True, figsize=(3*4, nrows*3), sharey=True)
    tckfmt = dict(style='sci', axis='y', scilimits=(0,0))
    # lgnarg = dict(bbox_to_anchor=(1.1, 0.9), frameon=False, ncol=1, 
    #               title=config['crossxi']['title'], fontsize=15)
    lgnarg = dict(ncol=3, frameon=False,
                  bbox_to_anchor=(0, 1.1, 3, 0.4), loc="lower left",
                  mode="expand", borderaxespad=0,  title=title)        
    xlim   = (-0.5, 10.5)
    ylim   = (-0.5, 5.5)
    
    xi_sys  = np.load(file_xisys, allow_pickle=True).item()['cross']

    def add_plot(filename, ax, color='b', label='none', linestyle='-',
                 xi_sys=xi_sys, errorbar=False, addauto=False, marker='None',
                factor=factor):    
        '''
            
        '''
        data     = np.load(filename, allow_pickle=True).item()
        xi_auto  = data['auto']['werr']# - d['auto']['dmean']**2
        xi_cross = data['cross']
        assert data['auto']['dmean'] < 1.0e-8

        for i in range(len(xi_cross)):
            xi_sys_auto  = xi_sys[i]['w'][0]/xi_sys[i]['w'][1] - xi_sys[i]['dmean']**2
            
            myd         = xi_cross[i]
            angle       = np.degrees(myd['t'][1:])            
            xi_cross_i  = factor*(myd['w'][0]/myd['w'][1]-myd['dmean1']*myd['dmean2'])**2/xi_sys_auto
            if errorbar:
                xi_error = factor*myd['werr']**2/xi_sys_auto
                ax[i].errorbar(angle, xi_cross_i, yerr=xi_error, 
                              color=color, label=label,
                              linestyle=linestyle, marker=marker)
            else:
                ax[i].plot(angle, xi_cross_i, color=color, label=label, 
                          linestyle=linestyle, marker=marker)
            if addauto:
                ax[i].fill_between(angle, 0, factor*xi_auto, color='grey', alpha=0.2) #label=r'Error on $\omega^{g,g}$'

    fig, ax = plt.subplots(**pltarg)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    ax = ax.flatten()
    for i, fi in enumerate(files_names):
        if i == 0:
            addauto=True
        else:
            addauto=False
        add_plot(files_names[i], ax, color=colors[i],
                 label=labels[i],
                 linestyle=linestyles[i], addauto=addauto)
        
    for i in range(xi_sys.shape[0]):
        if i ==6:ax[i].set_ylabel(r'$10^{}\times(\omega^{})^{}/\omega^{}$'\
                                  .format('{%.1f}'%np.log10(factor), '{~g,s}','2', '{s,s}'))
        if i ==16:ax[i].set_xlabel(r'$\theta$[deg]')
        ax[i].text(0.65, 0.75, r'%s'\
                          %config['crossxi']['xticks'][i],
                           transform=ax[i].transAxes, fontsize=15)
        ax[i].ticklabel_format(**tckfmt)
        ax[i].tick_params(axis='y', pad=1.0)
        ax[i].set_xlim(*xlim)
        ax[i].set_ylim(*ylim) 
        
    ax[0].legend(**lgnarg)
    #f.delaxes(a[17])

    if config['crossxi']['saveto'] is not None:
        plt.savefig(config['crossxi']['saveto'], bbox_inches='tight')
    plt.show()            
        
def plot_clxi(filen, filen2, ax, labels, hold=False, bins=None, saveto=None):
    '''
    
    Example :
    
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), sharex='col')
    ax = ax.flatten()
    # plt.subplots_adjust(hspace=0.0)

    for i,cap in enumerate(['ngc', 'sgc']):
        CAP = cap.upper()
        # #'v6_wnn_p', 'v6_wnnz_p', 'v6_wnnz_abv2', 'v6_wnn_abv2', 'v6_zelnet'],
        # #'nn (plain)', 'nn-z (plain)', 'nn-z', 'nn', 'elastic-z'],
        kw     = dict(lb    = ['v6_wosys','v6', 'v6_wnnz_abv2'],
                      lt    = ['No Correction', 'systot', 'nn-z'], 
                      c     = ['k', 'k', 'purple', 'royalblue', 'crimson', 'olive', 'g', 'darkorange'],
                      ls    = 2*['-', '--', '-', '-.', '--', '-'],
                      mk    = 2*['.', 'o', '^', 'x'],
                      title = CAP)    
        filen  = lambda l:'/home/mehdi/data/eboss/v6/results_'+cap+'.all/clustering/cl_'+CAP+'_'+l+'.npy'
        filen2 = lambda l:'/home/mehdi/data/eboss/v6/results_'+cap+'.all/clustering/xi_'+CAP+'_'+l+'.npy'
        dataviz.plot_clxi(filen, filen2, [ax[2*i], ax[2*i+1]], kw, hold=True)
    plt.show() 
    
    '''
    if bins is None:bins=np.logspace(np.log10(0.9), np.log10(1030), 8)
    
    
    lb  = labels['lb']
    c   = labels['c']
    mk  = labels['mk']
    ls  = labels['ls']
    lt  = labels['lt']
    ttl = labels['title']
    
    #n  = len(lb)-1
    #fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    #plt.subplots_adjust(wspace=0.4)

    kw = dict(bins=bins)
    for i,lbi in enumerate(lb):
        cl = np.load(filen(lbi), allow_pickle=True).item()
        elb, clbe = binit_jac(cl['clerr']['cljks'], **kw)
        elb, clb  = binit(np.arange(cl['auto'].size), cl['auto'], **kw)
        #print(clb)
        ax[0].errorbar(elb, clb, yerr=clbe, marker='.', linestyle=ls[i], color=c[i], label=lt[i])

    ax[0].loglog()
    ax[0].set_ylim(1.e-7, 1.e-2)
    ax[0].set_xlabel(r'$\ell$')
    ax[0].set_ylabel(r'$C_{\ell}$')

    fc = 1.e2
    for i, xii in enumerate(lb):
        d = np.load(filen2(xii), allow_pickle=True).item()['auto']
        t  = 0.5*np.degrees(d['t'][1:]+d['t'][:-1])
        xi = fc*(d['w']-d['dmean']*d['dmean'])

        xierr=d['werr']
        #print(xi)
        ax[1].errorbar(t, xi, yerr=fc*xierr, linestyle=ls[i], color=c[i], label=lt[i])
        #ax[0].legend(bbox_to_anchor=(0, 1.1, 2, 0.4), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=3, frameon=False)    
        ax[1].set_xlabel(r'$\theta [deg]$')
        ax[1].set_ylabel(r'$10^{2}\times \omega(\theta)$')
        ax[1].set_xlim(-0.2, 10)
        ax[1].set_ylim(-0.5, 2.0)
        ax[1].grid(True)
        ax[1].legend(bbox_to_anchor=(1.1,1.0), title=ttl)        
    if saveto is not None:plt.savefig(saveto, bbox_inches='tight')
    if not hold:plt.show()


        
def plot_nnbar(nnbars, title=None, axes=[i for i in range(17)],
               figax=None, annot=False, lb=None, cl=None, err=False,
              hold=False, lgannot=False):
    '''
    All:
        dataviz.plot_nnbar(['/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z0.8.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.1.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.3.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.5.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.7.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.9.npy'
                              ],
                       title='w/ systot',
                    lb=['0.8<z<1.1', '1.1<z<1.3', '1.3<z<1.5', 
                        '1.5<z<1.7', '1.7<z<1.9', '1.9<z<2.2'])

    Single:
        fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
        dataviz.plot_nnbar(['/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z0.8.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.1.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.3.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.5.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.7.npy',
                               '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_v6_z1.9.npy'
                              ],
                       title='w/ systot',
                       lb=['0.8<z<1.1', '1.1<z<1.3', '1.3<z<1.5', 
                            '1.5<z<1.7', '1.7<z<1.9', '1.9<z<2.2'],
                       axes=[6], figax=(fig, [ax]),
                      cl=plt.cm.jet)    
    Two:
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(16, 18),
                          sharey=True, sharex='col')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax=ax.flatten()
    for i, k in enumerate(['v6_wosys', 'v6', 'v6_wnnz_p']):    
        h = False if i==2 else True
        lg= False if i==0 else True
        dataviz.plot_nnbar(['/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_'+k+'_z0.8.npy',
                           '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_'+k+'_z1.1.npy',
                           '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_'+k+'_z1.3.npy',
                           '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_'+k+'_z1.5.npy',
                           '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_'+k+'_z1.7.npy',
                           '/home/mehdi/data/eboss/v6/results_ngc.all/clustering/nnbar_NGC_'+k+'_z1.9.npy'
                          ],
                           title=k,
                           lb=['0.8<z<1.1', '1.1<z<1.3', '1.3<z<1.5', 
                                '1.5<z<1.7', '1.7<z<1.9', '1.9<z<2.2'],
                           axes=[8, 13], figax=(fig, [ax[2*i], ax[2*i+1]]),
                          cl=plt.cm.jet, hold=h, lgannot=lg)                          
    '''
    nrows = len(axes)// 4
    if len(axes)%4!=0:nrows +=1
    if figax is None:
        fig, ax = plt.subplots(ncols=4, nrows=nrows, figsize=(20, 4*nrows), sharey=True)
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        ax = ax.flatten()    
    else:
        fig, ax = figax
    #if title is not None:fig.suptitle(title)     
    if title is not None:ax[0].text(0.1, 0.9, title, transform=ax[0].transAxes)
    if cl is None:cl=plt.cm.Blues
    chi2 = lambda y1, y2, ye: np.sum((y1-y2)*(y1-y2)/(ye*ye))/ye.size
    #lt = ['lin', 'NN+Ablation', 'NN', 'quad', 'No Correction']
    #cl = ['r', 'b', 'k', 'g', 'purple']
    ls = ['--', ':', '-', '-.', '-']
    n  = len(nnbars)
    for j,nnbar_i in enumerate(nnbars):
        nnbar = np.load(nnbar_i, allow_pickle=True).item()
        lt = lb[j]
        #lt = '_'.join([nnbar_i.split('/')[-1].split('_')[1], nnbar_i.split('/')[-1].split('_')[-1][:-4]])
        #print(lt, end=' ')
        c  = cl((j+2)/(n+1))        
        chi2tot = 0.0
        m = 0
        for ji,i in enumerate(axes):
            mynnb = nnbar['nnbar'][i]
            x     = 0.5*(mynnb['bin_edges'][1:]+mynnb['bin_edges'][:-1])
            y     = mynnb['nnbar']
            ye    = mynnb['nnbar_err']
            chi2i = chi2(y, 1.0, ye)
            chi2tot += chi2i
            if err:
                ax[ji].errorbar(x, y, yerr=ye, marker='.', color=c, label=lt)
            else:
                ax[ji].plot(x, y, marker='.', color=c, label=lt)
                ax[ji].fill_between(x, 1-ye, 1+ye, color=c, alpha=0.2)
            if annot:
                ax[ji].text(0.05+0.15*j, 0.9, '%.1f'%chi2i, 
                        transform=ax[ji].transAxes, color=c)
            if j==n-1:
                if len(axes)<2:
                    if not lgannot:ax[0].legend(bbox_to_anchor=(1.3, 1.0))
                else:
                    if not lgannot:ax[0].legend(**dict(ncol=4,frameon=False,
                                 bbox_to_anchor=(0, 1.1, 3, 0.4), loc="lower left",
                                 mode="expand", borderaxespad=0, fontsize=20))
                #ax[0].set_xlim(1, 3)
                ax[ji].set_ylim(0.9, 1.1)
                ax[ji].set_xlabel(nnbar['xlabels'][i])
                ax[ji].grid()
                if ji==0:ax[ji].set_ylabel(r'$\frac{n}{\overline{n}}$')
            m +=1
        print('%.1f  %d  %.1f'%(chi2tot, m, chi2tot/m))
        if annot:ax[-1].text(0.05, 0.9-0.1*j, '%.1f %d %.1f'%(chi2tot, m, chi2tot/m),
                             color=c, transform=ax[-1].transAxes)
    if not hold:plt.show()                            