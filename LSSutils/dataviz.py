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