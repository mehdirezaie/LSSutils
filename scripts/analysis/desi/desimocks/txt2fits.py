'''
    Code to change a txt file to fits
'''
import numpy  as np
import fitsio as ft
import sys

from time import time


finput  = sys.argv[1]
#foutput = finput.replace('.txt', '.fits')
foutput = sys.argv[2]

print('Going to read %s  and write %s'%(finput, foutput))

t0 = time()


x,y,z,z_rsd = np.loadtxt(finput).T
outdata = np.zeros_like(x, dtype=[('x','f8'), 
                                  ('y','f8'),
                                  ('z', 'f8'),
                                  ('z_rsd', 'f8')])
outdata['x']     = x
outdata['y']     = y
outdata['z']     = z
outdata['z_rsd'] = z_rsd
ft.write(foutput, outdata, clopper=True)

t1 = time()
assert x.size == outdata.size
print('done in {} sec'.format(t1-t0))
