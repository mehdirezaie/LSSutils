""" Nbodykit Json to TXT
"""
def json2txt(json, txt):
    
    dpk = nb.ConvolvedFFTPower.load(json)
    
    with open(txt, 'w') as f:

        # add header
        for attr, val in dpk.attrs.items():
            f.write(f'# {attr:21s}: {val}\n')

        nbin = dpk.poles['k'].size           
        k_ = dpk.poles.edges['k']
        ka_ = dpk.poles['k']
        p0_ = dpk.poles['power_0'].real
        p2_ = dpk.poles['power_2'].real
        
        # write spectra
        f.write('# kmin kavg P0 P2\n')        
        for i in range(nbin):
            f.write(f"{k_[i]:.6f} {ka_[i]:.6f} {p0_[i]:.6f} {p2_[i]:.6f}\n")
            
    plt.figure()
    plt.plot(ka_, p0_)
    plt.xscale('log')
    plt.show()
    