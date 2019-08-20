import matplotlib.pyplot as plt
import numpy  as np
import healpy as hp
from matplotlib.projections.geo import GeoAxes
from matplotlib import cm
from matplotlib.colors import ListedColormap


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
            width=7, figax=None, colorbar=False, cmap=plt.cm.bwr, **kwargs):
    '''
        (c)Andrea Zonca, https://github.com/zonca/paperplots 
        modified by Mehdi Rezaie for galaxy counts
        
        Matplotlib has higher freedom than healpy.mollview

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
        
    '''    
    nside     = hp.npix2nside(len(m))
    rotatedeg = np.degrees(rotate)    
    ysize     = xsize/2.                       # ratio is always 1/2

    
    # galactic plane
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
    else:
        fig = figax[0]
        ax  = figax[1]

    # rasterized makes the map bitmap while the labels remain vectorial
    # flip longitude to the astro convention
    image = ax.pcolormesh(longitude, latitude, rot(grid_map, rotatedeg), 
                           vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)
    ax.scatter(phi_cl-2/3*np.pi, np.pi/2-theta_cl, 2, color='r')
    ax.scatter(phi_cl+4/3*np.pi, np.pi/2-theta_cl, 2, color='r')
    # graticule
    ax.set_longitude_grid(60)
    ax.set_latitude_grid(30)
    ax.xaxis.set_major_formatter(ThetaFormatterShiftPi())

    # colorbar
    if colorbar:
        #cax = plt.axes([.9, 0.2, 0.01, 0.6])  # vertical
        cax = plt.axes([0.2, 0.0, 0.6, 0.01])  # horizontal
        cb  = fig.colorbar(image, cax=cax, label=unit, 
                           shrink=0.6, pad=0.05, ticks=[vmin, vmax], orientation='horizontal', extend='both')        
        #cb = fig.colorbar(image, orientation='horizontal', shrink=.6, pad=0.05, ticks=[vmin, vmax])
        #cb.ax.xaxis.set_label_text(unit)
        cb.ax.xaxis.labelpad = -8
        # workaround for issue with viewers, see colorbar docstring
        cb.solids.set_edgecolor("face")

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

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





def hyper_params_data(files1=None, tl=['eBOSS QSO V6'], verbose=False, labels=None):
    def get_all(ablationlog):
        d = np.load(ablationlog, allow_pickle=True).item()
        indices = None
        for il, l in enumerate(d['validmin']):
            m = (np.array(l) - d['RMSEall']) > 0.0
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
                    ax.scatter(i, axes[i][j], c=[colors[j]], marker='o', **kw)    
        

    #labels = ['EBV', 'lnHI', 'nstar']
    #labels += [''.join((s,'-',b)) for s in ['depth', 'seeing', 'skymag', 'exptime', 'mjd']\
    #           for b in 'rgz']
    #labels = ['sky_g', 'sky_r', 'sky_i', 'sky_z', 
    #    'depth_g', 'depth_r', 'depth_i','depth_z',
    #    'psf_g','psf_r', 'psf_i', 'psf_z',
    #    'w1_med', 'w1_covmed',
    #    'star_density', 'ebv', 'airmass']
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(6, 4))
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
    #plt.savefig('./figs/hyper-params_data.pdf', bbox_inches='tight')
    plt.show()






def ablation_plot(filename, odir, labels=False):    
    import seaborn as sns
    ab1 = np.load(filename, allow_pickle=True).item()
    #
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
                matric[i,j] = (matric_dict['%d-%d'%(i,sys_i)][0]/ab1['RMSEall'])-1.#-ab1['baselineRMSE']
    matric *= 1.e4
    #bands = ['r','g','z']
    #labels = ['ebv','lnHI','nstar']
    #labels += ['depth-'+b for b in bands]
    #labels += ['seeing-'+b for b in bands]
    #labels += ['airmass-'+b for b in bands]
    #labels += ['skymag-'+b for b in bands]
    #labels += ['exptime-'+b for b in bands]
    #labels += ['mjd-'+b for b in bands]

    xlabels = [labels[j] for j in FEAT]
    mask = ~np.zeros_like(matric, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = False
    mask[matric==0.0] = False
    vmin = np.minimum(np.abs(np.min(matric)), np.abs(np.max(matric))) #* 0.1
    print(np.max(matric), np.min(matric))
    # Set up the matplotlib figure
    f, ax = plt.subplots()
    #plt.title('Correlation Matrix of DR5')
    # Generate a custom diverging colormap
    kw = dict(mask=~mask, cmap=plt.cm.seismic, xticklabels=xlabels, #PRGn_r,get_cmap('PRGn_r', 20)
               yticklabels=xlabels[::-1], 
               vmax=20, center=0.0,#vmin=-1.*vmin, vmax=vmin, center=0.0,
               square=True, linewidths=.5, 
               cbar_kws={"shrink": .5, 
               "label":r'$10^{4} \times \delta$RMSE', "extend":"max"})
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matric, **kw)
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticks([])
    ax.xaxis.tick_top()
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
    fname = filename.split('/')[-1][:-8]  # drop .log.npy
    #ou = ''.join([filename[:-4], '.pdf']) # drop .npy
    #ou = ''.join([odir, fname, '.pdf'])
    #print('save ... ', ou)
    #plt.savefig('./figs/dr7ablation.pdf', bbox_inches='tight') 
    plt.show()


