
import numpy as np

def get_veff(cap='NGC', P0=6000.):
    
    d = np.loadtxt(f'/home/mehdi/data/eboss/data/v7_2/nbar_eBOSS_QSO_{cap}_v7_2.dat')
    
    z = d[:, 1]
    nz = d[:, 3]
    dvz = d[:, 5]
    #wfkp = lambda nz: 1./(1.+nz*P0)

    veff = (nz*P0 / (1. + nz*P0))**2*dvz *1.0e-9 # (Gpc/h)^3

    zlim = {'main':(z > 0.8) & (z < 2.2),
            'highz':(z > 2.2) & (z < 3.5)}

    for s,lim in zlim.items():
        print(f'{cap}, {s:5s}, Veff = {veff[lim].sum():.3f} (Gpc/h)^3 with P0={P0:.1f}')    
        
        
for cap in ['NGC', 'SGC']:
    for p0 in [6000., 10000., 20000.]:
        get_veff(cap, p0)
    print(20*'-')
    
"""
NGC, main , Veff = 0.120 (Gpc/h)^3 with P0=6000.0
NGC, highz, Veff = 0.009 (Gpc/h)^3 with P0=6000.0
NGC, main , Veff = 0.294 (Gpc/h)^3 with P0=10000.0
NGC, highz, Veff = 0.024 (Gpc/h)^3 with P0=10000.0
NGC, main , Veff = 0.888 (Gpc/h)^3 with P0=20000.0
NGC, highz, Veff = 0.086 (Gpc/h)^3 with P0=20000.0
--------------------
SGC, main , Veff = 0.065 (Gpc/h)^3 with P0=6000.0
SGC, highz, Veff = 0.005 (Gpc/h)^3 with P0=6000.0
SGC, main , Veff = 0.161 (Gpc/h)^3 with P0=10000.0
SGC, highz, Veff = 0.013 (Gpc/h)^3 with P0=10000.0
SGC, main , Veff = 0.495 (Gpc/h)^3 with P0=20000.0
SGC, highz, Veff = 0.045 (Gpc/h)^3 with P0=20000.0
"""