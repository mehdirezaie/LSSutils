#
#	This program is aimed at finding the multiple power spectra
#	last modified by Mehdi on Dec 18, 2015
#
import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import numpy.polynomial.legendre as npl
#
# reading the input file, that includes k and pk
#
kh,pk = np.loadtxt('camb_pk.txt',unpack=True)

# Gauss Legendre Quadrature points and weights
x,wx = npl.leggauss(5)


# Legendre Polynomial
l = 0
c = np.zeros(5)
c[l] = 1.0


px = npl.legval(x,c)



redshift = .55
bias = 2.0
omegm0 = 0.31	#Omega_M = .31    Omega_M h^2 = 0.12 where h = 0.67
omegm = (omegm0*((1.0+redshift)**3))/(omegm0*((1.0+redshift)**3)+1.0-omegm0)
beta = (1.0/bias)*((omegm)**0.6)
print 'Omega(z='+str(redshift)+') =',omegm,' Beta =',beta


integral1 = 0.0
for i in range(np.size(x)):
	integral1 += wx[i]*px[i]*((1.0+beta*x[i]*x[i])**2)

integral1 *= bias*bias*(2.0*float(l)+1.0)*0.5

pk *= integral1


kh1,pk0,pk2,pk4 = np.loadtxt('ps1D_DR12CMASSLOWZ-NGC-V6C_242_454_487_120.dat',\
                              skiprows = 30,usecols = (0,2,3,4),unpack= True)



fig1,fig2 = plt.loglog(kh,pk,'b+',kh1,pk0,'r-')
plt.legend((fig1,fig2),('Camb','Florian'))
plt.title("Monopole l = "+str(l))
plt.savefig('camb_simulation.pdf',format='pdf')
plt.show()
plt.close()
