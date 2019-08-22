'''

   Code to compute the 2pc for a Simulation Box

'''
import sys
import numpy as np
import nbodykit.lab as nb               # Import Nbodykit
from nbodykit import setup_logging
from time import time


setup_logging("info")                   # Info


from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size


#
#  I/O 
#
if rank == 0:
    input_name  = sys.argv[1]
    output_name = sys.argv[2]
else:
    input_name  = None
    output_name = None
input_name  = comm.bcast(input_name,  root=0)
outout_name = comm.bcast(output_name, root=0) 


# read data
catalog = nb.FITSCatalog(input_name)
for col in ['x', 'y', 'z']:
    if not col in catalog.columns:raise RuntimeError('%s not available'%col)

catalog['Position'] = np.column_stack([catalog['x'], catalog['y'], catalog['z']])



#
# Inputs for Corrfunc
#
mode  = '2d'      # r, mu
nbin  = 200       
nmu   = 120
rmin  = 0.0
rmax  = 200. 
box   = 1000      # 1000 Mpc/h 


edges   = np.linspace(rmin, rmax, nbin+1) 

if rank==0:t0=time()
results = nb.SimulationBox2PCF(mode, catalog, edges, Nmu=nmu,
                               periodic=True, BoxSize=box, los='z', 
                               weight='Weight', position='Position', 
                               show_progress=True)
if rank==0:print('took {} sec'.format(time()-t0))
results.save(output_name)
