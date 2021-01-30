""" Compute mitigation bias """
import sys
import matplotlib.pyplot as plt
import numpy as np
import nbodykit.lab as nb
from tqdm import tqdm

path = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
pkname_fn = lambda p, c, m, n, i, ix:f'{p}spectra_{c}_{m}_mainhighz_{n}_v7_{i}_{ix:04d}_main.json'    
    
def solve(x, y):
    n = len(x)
    sumy = np.sum(y)
    sumx = np.sum(x)
    sumx2 = np.sum(x*x)
    sumxy = np.sum(x*y)

    b = (sumy*sumx2-sumx*sumxy)/(n*sumx2-sumx*sumx)
    m = (n*sumxy - sumx*sumy)/(n*sumx2-sumx*sumx)
    return m, b

def readpk(*params):
    filename = pkname_fn(*params)
    d = nb.ConvolvedFFTPower.load(filename)
    return (d.poles['k'], d.poles['power_0'].real - d.attrs['shotnoise']) # 

class Spectra:
    
    def __init__(self, cap='NGC', nside='512'):    
        self.cap = cap
        self.nmocks = 1000 if cap=='NGC' else 999
        self.nside = nside

        
    def load(self, method, iscont):        
        pks = []
        for ix in tqdm(range(1, self.nmocks+1)):
            k0, p0 = readpk(path, self.cap, 'knownsystot', self.nside, '0', ix)
            k1, p1 = readpk(path, self.cap,  method, self.nside, iscont, ix)
            pks.append([p0, p1, p1-p0])

        assert np.array_equal(k0, k1)
        self.k = k0
        self.pks = np.array(pks)
        self.method = method
    
    def get_params(self, vs=1, saveto=None, plot=False):
        assert (vs in [0, 1])
        
        legend = {0:'Biased Truth',
                  1:'Unbiased mitig.'}
        
        self.ms = []
        self.bs = []
        
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(18, 8), sharey=True)
        ax = ax.flatten()
        
        for j in tqdm(range(self.pks.shape[2])):
            bins_p = np.percentile(self.pks[:, vs, j], [0, 20, 40, 60, 80, 100])
            bins_p[-1] += 1.0e-4
            ix = np.digitize(self.pks[:, vs, j], bins=bins_p)
            x_ = []
            y_ = []
            for i in ix:
                s_i = ix == i
                x_.append(np.median(self.pks[s_i, vs, j], axis=0))
                y_.append(np.median(self.pks[s_i, 2, j], axis=0))

            x_ = np.array(x_)
            y_ = np.array(y_)

            m,b  = solve(x_, y_)            

            self.ms.append(m)
            self.bs.append(b)

            if j < 6:    
                ax[j].scatter(self.pks[:, vs, j], self.pks[:, 2, j], 1.0, color='k', alpha=0.2)
                ax[j].scatter(x_, y_, label=f'median (mean=%.1f)'%np.mean(y_), marker='s', color='r')
                ax[j].plot(bins_p, m*bins_p + b, 'r-')
                print(self.k[j], m, b) 

                #ax[j].set(ylim=(0., 4), xlabel=xlabel[vs], ylabel='Pcont,mitig/Ptrue')
                ax[j].set(xlabel='Px,mitig', ylabel='Pcont,mitig-Ptrue')#ylim=(0., 10),
                ax[j].legend(title=f'k={self.k[j]:.4f}')
                
            
        if saveto is not None:
            np.savetxt(saveto, np.column_stack([self.k, self.ms, self.bs]), 
                       header=f"kavg [h/Mpc], slope, intercept [Mpc/h]**3 ({self.cap}, {self.nside}, {self.method})",
                                                        fmt='%.8f')
            
        if plot:
            plt.rc('font', size=15)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            ax1.plot(self.k, self.ms, 'k-', marker='.', mfc='w', lw=0.5)
            ax2.plot(self.k, self.bs, '--', color='crimson', marker='.', mfc='w', )


            ax1.set_ylabel(r'Slope')
            ax1.text(0.4, 0.9, f'{self.cap} ({self.method}-{self.nside})', transform=ax1.transAxes)
            ax1.tick_params(axis='both', direction='in')
            ax1.set_xlabel('k [h/Mpc]')
            #ax1.set_ylim()

            ax2.set_ylabel(r'Intercept [Mpc/h]$^{3}$', color='crimson')
            ax2.tick_params(axis='y', direction='in', colors='crimson', which='both')
            ax2.spines['right'].set_color('crimson')
            ax2.set_xscale('log') 
            if saveto is not None:
                fig.savefig(saveto.replace('.txt', '_params.pdf'), bbox_inches='tight')

            good = self.k > 0.0#015 # ignore the first k bin
            Ptrue_ = self.pks[:,0,good]
            Ptrue_mean = Ptrue_.mean(axis=0)            
            pk_err = np.std(Ptrue_, axis=0) / np.sqrt(self.nmocks)

            Pmitig_mean = self.pks[:,1,good].mean(axis=0)
            if vs==1:
                Pcorrec = (1-np.array(self.ms))*self.pks[:, 1,:]-np.array(self.bs)
            else:
                Pcorrec = (1 + np.array(self.ms))*self.pks[:, 0,:] + np.array(self.bs)
            
            fig, ax = plt.subplots(nrows=2, figsize=(6, 8), sharex=True)
            fig.subplots_adjust(hspace=0.0)

            ax[0].errorbar(self.k[good], Ptrue_mean, pk_err, marker='.', 
                           ls=':', color='k', capsize=3, zorder=-10)
            ax[0].plot(self.k[good], Pmitig_mean, '.b--',
                       self.k[good], Pcorrec[:, good].mean(axis=0), '.r-', mfc='w')
            #ax[0].fill_between(self.k[good], pkmin, pkmax, color='k', alpha=0.1, zorder=-10)

            ax[0].set(xscale='log', ylabel=r'P0 [Mpc/h]$^{3}$')

            ax[1].plot(self.k[good], Pmitig_mean/abs(Ptrue_mean), 'b--',
                       self.k[good], Pcorrec[:, good].mean(axis=0)/abs(Ptrue_mean), 'r-')
            ax[1].axhline(1.0, ls=':', color='k')
            ax[1].legend(['Mitigated', legend[vs], 'Truth'],
                        title=f'{self.cap} ({self.method}-{self.nside})')
            
            ax[1].errorbar(self.k[good], np.ones(good.size), pk_err/abs(Ptrue_mean), ls='None', color='k', capsize=3, zorder=-10)
            ax[1].set(xscale='log', xlabel='k [h/Mpc]', ylabel=r'P0/|Ptruth|', ylim=(0., 2.0))#, xlim=(0.01, 0.3), ylim=(0.9, 1.1)) 
            if saveto is not None:
                fig.savefig(saveto.replace('.txt', '_power.pdf'), bbox_inches='tight')            
                
if __name__ == '__main__':   
    if len(sys.argv) != 4:
        print('run with $> python [name of file].py cap nside method')
        sys.exit()
    cap = sys.argv[1]
    nside = sys.argv[2]
    method = sys.argv[3]

    output_name = ''.join([path, f'pk_params_{cap}_{nside}_{method}.txt'])


    sp = Spectra(cap, nside)
    sp.load(method, '1')
    sp.get_params(saveto=output_name, plot=True)                
