#
#	This program is aimed at finding the multiple power spectra
#	last modified by Mehdi on Dec 18, 2015
#
import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import numpy.polynomial.legendre as npl
import scipy.special as scs


#
#	Eisenstein 1997
#	d1(v) = D(a)/a
#
def d1(v):
    beta = np.arccos((v+1. - np.sqrt(3.)) / (v+1. + np.sqrt(3.)))
    sin75 = np.sin(75. * np.pi/180.)
    sin75 = sin75*sin75
    result = (5./3.) * (v) * (((3.**0.25) * (np.sqrt(1. + v**3.)) * (scs.ellipeinc(beta, sin75)
             - (1. / (3.+np.sqrt(3.))) * scs.ellipkinc(beta, sin75)))
             + ((1. - (np.sqrt(3.)+1.)*v*v) / (v+1. + np.sqrt(3.))))
    return result
#
#	f = dln(D(a))/dln(a)
#
def growthfactor(a, omega0):
    v = scs.cbrt(omega0/(1.-omega0)) / a
    return (omega0 / (((1.-omega0)*a**3)+omega0)) * ((2.5/d1(v)) - 1.5)


#
#	print growth factor
#
#z = np.arange(0.1,10.,0.02)
#a = 1./(1.+z)
#gf= growthfactor(a,0.3)

#plt.ylim(0.1,1.1)
#fig1,=plt.loglog(z,gf,'r--')
#plt.legend((fig1,),('growth factor',),loc = 4)
#plt.xlabel('z')
#plt.savefig("gf.pdf",format = 'pdf')
#plt.show()
#plt.close()



#
#	print growth function
#
#v = np.arange(0.1,10.,0.02)
#d1v = d1(v)
#plt.ylim(0.1,1.1)
#fig1,=plt.loglog(v,d1v,'--')
#plt.legend((fig1,),('d1(v)',),loc=4)
#plt.xlabel('v')
#plt.savefig("d1.pdf",format = 'pdf')
#plt.show()











# Gauss Legendre Quadrature points and weights
x, wx = npl.leggauss(500)


# Legendre Polynomial
l = 0
c = np.zeros(5)
c[l] = 1.0


px = npl.legval(x, c)



redshift = .55
bias = 2.0
omega0 = 0.274
a = 1. / (1. + redshift)
beta = (1.0 / bias) * (growthfactor(a, omega0))
print 'Omega(z=%f) = %f, beta = %f'%(redshift,omega0,beta)


integral1 = 0.0
for i in range(np.size(x)):
	integral1 += wx[i] * px[i] * ((1.0 + beta * x[i]*x[i])**2)

integral1 *= (2.0*float(l) + 1.0)*0.5



# CAMB input
kh_camb,pk_camb = np.loadtxt('pk_kzp55.txt',usecols = (0,1),unpack= True)
pk_camb *= integral1*(bias**2)

# Florian input
#kh1,pk0,pk2,pk4 = np.loadtxt('ps1D_DR12CMASSLOWZ-NGC-V6C_242_454_487_120.dat',skiprows = 30,usecols = (0,2,3,4),unpack= True)



# Hee-Jong
#kh_hj,pk_hj = np.loadtxt('pkr.mA00_hodfit.gals.dat',usecols = (0,1),unpack= True)
kh_hj,pk_hj = np.loadtxt('pks.A00_hodfit.galxs.dat',usecols = (0,1),unpack= True) #HJ output

#
# reading my code power spectrum (measured), that includes k and pk
#
kh,pk = np.loadtxt('k_powerv4.txt',usecols = (0,1),unpack=True)	#measured RS pk
kh_r,pk_r = np.loadtxt('k_powerv3.txt',usecols = (0,1),unpack=True)#expected RS pk from measure RS pk
pk_r *= integral1   #simulation does not need bias**2

#fig1,fig2 = plt.loglog(kh,pk,'--',kh_hj,pk_hj,'g--')
#plt.legend((fig1,fig2),('Simulation_Mehdi','HeeJong'))
#plt.xlabel('k(h.Mpc^-1)')
#plt.title ('Realspace_P(k)')
#plt.ylim(1e3,1e6)
#plt.savefig('real2.pdf',format = 'pdf')
#plt.show()

# plotting
fig1,fig2,fig3,fig4 = plt.loglog(kh,pk,'b--',kh_camb,pk_camb,'b-',kh_r,pk_r,'r--',kh_hj,pk_hj,'g--')
plt.legend((fig1,fig2,fig3,fig4),('Measured_Mehdi','Expected_Camb','Expected__from_Measured_Real_space','HJ'))
plt.xlabel('k(h.Mpc^-1)')
plt.title ('Monopole P0(k)')
plt.ylim(1e3,1e6)
plt.savefig('pk0_rs3.pdf',format = 'pdf')
plt.show()
sys.exit()
