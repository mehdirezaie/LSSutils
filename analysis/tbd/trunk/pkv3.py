# --------------------------------------------------------
#   version 3
#   This program calculates the matter power spectrum of a simulation box
#   assuming isotropy and homogeneity of space
#   uses: numpy.fft
#
#   (C) Mehdi
#       mehdirezaie1990@gmail.com
#
#   calculates P(k) from a N-body simulation
#
#
#   started         18-9-2015
#   last revised    29-12-2015
# --------------------------------------------------------


#   routines
import numpy as np
import matplotlib.pyplot as plt
import sys

#
#   reading input file that includes the simulation result
#
x,y,z = np.genfromtxt('../input_data/A00_hodfit.gal',comments='#',usecols=(0,1,2),unpack=True)

# for binary inputs uncomment this
# def readfn(fn, dt):
#     fp = open(fn, 'rb')
#     halos = np.fromfile(fp, dtype=dt)
#     return halos
# filename = 'halos_1.particles'
# ht = np.dtype([('x', np.float32, 3), ('v', np.float32, 3)])
# halos = readfn(filename, ht)
# x = halos['x'][:,0]
# y = halos['x'][:,1]
# z = halos['x'][:,2]
# x += 0.5
# y += 0.5
# z += 0.5
#

Length_box = 1500. 					# 1500 Mpc.h^-1
n_bins = 512						# num of mesh points

num_particles = len(x)					# total num of particles
avg_num_particle = float(num_particles)/float(n_bins)**3
normfactor = (Length_box**3)/(float(n_bins)**6)

print 'num of bins is :',n_bins
print 'number of particles is :',num_particles
print 'length of the box is :', Length_box
print 'average num of particles in each bin is :', avg_num_particle


#
#	binning data points in terms of integer indices k,l,m
#
n = np.zeros((n_bins,n_bins,n_bins))	# num of particles in each mess is stored in n array

for i in range(num_particles):		# multiplying coordinates by 512
    x[i] *= float(n_bins)		# to scale from (0,1) to (0,512)
    y[i] *= float(n_bins)
    z[i] *= float(n_bins)

for i in range(num_particles):		# binning particles
    ix = np.int(x[i])
    iy = np.int(y[i])
    iz = np.int(z[i])
    if (ix == 512):			# due to symmetry, particles get shifted
       ix = 0
    if (iy == 512):
       iy = 0
    if (iz == 512):
       iz = 0
    n[ix][iy][iz] += 1

#
# density fluctuation
#
n = [[[(float(k)/avg_num_particle)-1.0 for k in l] for l in m] for m in n]



#
# Shifted FFT of density fluctuation
#
n_k_space = np.fft.fftshift(np.fft.rfftn (n),axes=(0,1))


#
#	zero component frequency
#	real space: (N,N,N)  --- Fourier:(N,N,N/2)
#	after shift the zero comp. moves to (N/2,N/2,0), if we shift along z-axis
#
kx_zero = 0.5*float(n_k_space.shape[0])
ky_zero = 0.5*float(n_k_space.shape[1])
kz_zero = 0.0


scale_k = 2.*np.pi/Length_box		   # scale of k : 2\pi/L_max & k(i) = i*scale_k
delta_k = 1.0                              # in unit of 2\pi/L
k_min   = 0.0
k_max   = np.sqrt(3./4.)*float(n_bins)
n_bins_k_space = np.int((k_max-k_min)/delta_k)+1
power_k = np.zeros(n_bins_k_space)
n_modes = np.zeros(n_bins_k_space,dtype=np.int)

#
# making k-grids i*scale_k
#
k_vector = [(float(i))*scale_k for i in range(n_bins_k_space)]



#
# loop over k magnitudes, k = 0, dk, 2dk, 3dk, ... , k_max = 2\pi/delta_x
#
# loop over k-z: m from 0-N/2+1
for m in range(n_k_space.shape[2]):
    print m
    #loop over 2nd axis ie. k-y: l from 0-N
    for l in range(n_k_space.shape[1]):
        #loop over 1st axis ie. k-x: j from 0-N
        for j in range(n_k_space.shape[0]):
            # magnitude of k vector
            k = np.sqrt(((float(j)-kx_zero)**2)+((float(l)-ky_zero)**2)+((float(m)-kz_zero)**2))
            k_index = np.int(k/delta_k)
            #
            #	symmetries
            #
            if (j == 0 and l == 0 and m == 0):
                power_k[k_index] += 4.0*np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 4
            elif (j == 0 and l != 0 and m == 0):
                power_k[k_index] += 2.0*np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 2
            elif (j != 0 and l == 0 and m == 0):
                power_k[k_index] += 2.0*np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 2
            elif (j == 0 and l == 0 and m != 0):
                power_k[k_index] += 8.0*np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 8
            elif (j == 0 and l != 0 and m != 0):
                power_k[k_index] += 4.0*np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 4
            elif (j != 0 and l == 0 and m != 0):
                power_k[k_index] += 4.0*np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 4
            elif (j != 0 and l != 0 and m == 0):
                power_k[k_index] += np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 1
            elif (j != 0 and l != 0 and m != 0):
                power_k[k_index] += 2.0*np.abs(n_k_space[j][l][m])**2
                n_modes[k_index] += 2






power_k /= n_modes            # power = Sum over delta(k)^2/#ofModes
power_k *= normfactor         # Norm factor of P(k) : L^3/N^6
power_k[np.isinf(power_k)] = np.nan



#
#	output
#
np.savetxt('k_powerv3.txt',zip(k_vector,power_k,n_modes))
plt.title('P(k) vs k')
plt.ylim(1000,100000)
plt.loglog(k_vector,power_k)
plt.savefig('pk_k.pdf',format='pdf')
plt.close()
