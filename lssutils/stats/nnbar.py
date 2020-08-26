import os
import numpy as np
import logging

from lssutils import CurrentMPIComm
from lssutils.utils import split_NtoM

@CurrentMPIComm.enable
def get_meandensity(ngal, nran, mask, systematics,
                    selection_fn=None,
                    njack=20, nbins=8, 
                    columns=None,
                    comm=None):
    
    if columns is None:
        columns = ['sys-%d'%i for i in range(systematics.shape[1])]
        

    start, stop = split_NtoM(systematics.shape[1], comm.size, comm.rank)
            

    nnbar_list= []
    for i in range(start, stop+1):
        nnbar_i = MeanDensity(ngal, nran, mask, systematics[:,i],
                              nbins=nbins, selection=selection_fn)

        nnbar_i.run(njack=njack)
        nnbar_i.output['sys'] = columns[i] # add the name of the map
        nnbar_list.append(nnbar_i.output)

    comm.Barrier()
    nnbar_list = comm.gather(nnbar_list, root=0)
    
    if comm.rank==0:
        nnbar_list = [nnbar_j for nnbar_i in nnbar_list for nnbar_j in nnbar_i if len(nnbar_i)!=0]            
        return nnbar_list


class MeanDensity(object):
    """
    INPUTS:
    galmap, ranmap, mask,
    sysmap, bins, selection=None
    """

    logger = logging.getLogger('MeanDensity')

    @CurrentMPIComm.enable
    def __init__(self, galmap, ranmap, mask,
                       sysmap, nbins=8, selection=None, binning='equi-area',
                       percentiles=[0, 100], comm=None):
        #
        # inputs
        self.comm = comm
        self.galmap = galmap[mask]
        self.ranmap = ranmap[mask]
        self.sysmap = sysmap[mask]
        #
        # selection on galaxy map
        if selection is not None:
            if self.comm.rank==0:
                self.logger.info('apply selection mask on galaxy')
            self.galmap /= selection[mask]
        #
        # digitize
        self.sysl = [0 for k in range(3*nbins)]

        if binning == 'simple':

            smin, smax = np.percentile(self.sysmap, percentiles)
            bins = np.linspace(smin, smax, nbins+1)

            if self.comm.rank==0:
                self.logger.info(f'{nbins} {binning} bins from {smin} to {smax}')

            inds = np.digitize(self.sysmap, bins)

            for i in range(1,bins.size): # what if there is nothing on the last bin? FIXME
                self.sysl[3*i-3] = self.sysmap[np.where(inds == i)].tolist()
                self.sysl[3*i-2] = self.galmap[np.where(inds == i)].tolist()
                self.sysl[3*i-1] = self.ranmap[np.where(inds == i)].tolist()

        elif binning == 'equi-area':
            npts  = self.ranmap.size
            swtt  = self.ranmap.sum()/nbins  # num of randoms in each bin

            if self.comm.rank==0:
                self.logger.info(f'{swtt} randoms (area) in each bin')
            datat = np.zeros(npts, dtype=np.dtype([('ss', 'f8'), ('gs', 'f8'),
                                                   ('ws', 'f8'), ('rid', 'i8')]))
            datat['ss'] = self.sysmap*1
            datat['gs'] = self.galmap*1
            datat['ws'] = self.ranmap*1
            np.random.seed(123456)
            datat['rid'] = np.random.permutation(np.arange(npts))

            datas = np.sort(datat, order=['ss', 'rid'])
            ss, gs, ws = datas['ss'], datas['gs'], datas['ws']

            listg = []
            listr = []
            lists = []
            bins  = [ss[0]] # first edge is the lowest systematic
            j =  0
            swti = 0.0
            for i, wsi in enumerate(ws):
                swti += wsi
                listg.append(gs[i])
                listr.append(ws[i])
                lists.append(ss[i])
                if (swti >= swtt) or (i == npts-1):
                    swti  = 0.0
                    bins.append(ss[i])
                    self.sysl[3*j] = lists
                    self.sysl[3*j+1] = listg
                    self.sysl[3*j+2] = listr
                    lists = []
                    listg = []
                    listr = []
                    j += 1

            bins = np.array(bins)
            if self.comm.rank==0:
                self.logger.info('min sys : %.2f  max sys : %.2f'%(ss[0], ss[npts-1]))
                self.logger.info('num of pts : %d, num of bins : %d'%(i, j))
        else:
            raise ValueError('%s not among [simple, equi-area]'%binning)

        totgal = np.sum([np.sum(self.sysl[i]) for i in np.arange(1,3*nbins, 3)])
        totran = np.sum([np.sum(self.sysl[i]) for i in np.arange(2,3*nbins, 3)])
        self.avnden = totgal/totran
        self.bins = bins
        if self.comm.rank==0:
            self.logger.info(f'mean nbar {self.avnden}')

    def run(self, njack=20):
        sl = []
        ml = []
        nl = []
        bl = []
        for i in range(0, 3*self.bins.size-3, 3):
            bavg = 0.0
            ng   = 0.0
            std  = 0.0
            npix = 0.0
            for j in range(0,len(self.sysl[i])):
                bavg += self.sysl[i][j]*self.sysl[i+2][j]
                ng += self.sysl[i+1][j]
                npix += self.sysl[i+2][j]
            if npix == 0.0:
                bl.append(np.nan)
                ml.append(np.nan)
                nl.append(np.nan)
                sl.append(np.nan)
                continue

            mean = ng/npix/self.avnden
            bl.append(bavg/npix)
            ml.append(mean)
            nl.append(npix)

            if (len(self.sysl[i]) < njack) or (njack == 0):  # use the typical std if njack is 0
                for k in range(0,len(self.sysl[i])):
                    std += (self.sysl[i+1][k]/self.sysl[i+2][k]/self.avnden-mean)**2.
                std = np.sqrt(std)/(len(self.sysl[i])-1.)
            else:
                jkf = len(self.sysl[i])//njack
                for k in range(0,njack):
                    ng   = 0
                    npix = 0
                    minj = jkf*k
                    maxj = jkf*(k+1)
                    for j in range(0,len(self.sysl[i])):
                        if j < minj or j >= maxj:
                            ng   += self.sysl[i+1][j]
                            npix += self.sysl[i+2][j]
                    mj = ng/npix/self.avnden
                    std += (mj-mean)*(mj-mean)
                std = np.sqrt((njack-1.)/float(njack)*std)
            sl.append(std)
        #
        # area
        npixtot   = self.ranmap.size
        nrantot   = self.ranmap.sum()
        area1pix  = 1 #nside2pixarea(self.nside, degrees=True)
        npix2area = npixtot*area1pix/nrantot
        #
        # prepare output
        output   = {}
        output['nnbar'] = np.array(ml)
        output['area'] = np.array(nl) * npix2area
        output['nnbar_err'] = np.array(sl)
        output['bin_edges'] = self.bins
        output['bin_avg'] = np.array(bl)
        attrs = {}
        attrs['njack'] = njack
        attrs['nbar'] = self.avnden
        attrs['npix2area'] = npix2area

        output['attr'] = attrs
        self.output = output

    def save(self, path_output):
        
        dir_output = os.path.dirname(path_output)
        if not os.path.exists(dir_output):
            self.logger.info(f'creating {dir_output}')
            os.makedirs(dir_output)
            
        self.logger.info(f'writing the output in {path_output}')
        np.savez(path_output, **self.output)