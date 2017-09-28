# Convert 3D-PDR input files to 3D-RT input files

import numpy as np
import sys


filename = str(sys.argv[1])

data = np.loadtxt(filename)

ngrid = np.shape(data)[0]


# Rotate the gid such that the 1D line coincides with a HEALPix ray

x = 0.527046*data[:,0]
y = 0.527046*data[:,0]
z = 0.666667*data[:,0]

vx = np.zeros(ngrid)
vy = np.zeros(ngrid)
vz = np.zeros(ngrid)

density = data[:,3]


input_data = np.stack((x, y, z, vx, vy, vz, density), axis=1)
input_filename = filename + "_conv.txt"
np.savetxt(input_filename, input_data, fmt='%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')

print("Grid (1D) created with", ngrid, "grid points!")
