#
#   script to plot the power spectra
#

import matplotlib.pyplot as plt
import numpy as np

k,pk = np.loadtxt('k_powerv3.txt',usecols = (0,1),unpack=True)
k2,pk2 = np.loadtxt('psN_512.dat',usecols = (0,1),unpack=True)


fig1,fig2 = plt.loglog(k,pk,'b-',k2,pk2,'r--')
plt.legend((fig1,fig2),('Mehdi','Jerry'))
plt.ylim(1000,100000)
plt.savefig('fig1.pdf',format='pdf')
#plt.show()
plt.close()
