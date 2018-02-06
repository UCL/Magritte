# Convert 3D-PDR input files to Magritte input files

import numpy as np
import sys


filename = str(sys.argv[1])

data   = np.loadtxt(filename)
ncells = np.shape(data)[0]


ID = np.zeros(ncells)

x = data[:,0]
y = data[:,1]
z = data[:,2]

vx = np.zeros(ncells) #data[:,0]*1.0E5
vy = np.zeros(ncells) #data[:,0]*1.0E5
vz = np.zeros(ncells) #data[:,0]*1.0E5

density = data[:,3]


# Save converted input file
input_data     = np.stack((ID, x, y, z, vx, vy, vz, density), axis=1)
input_filename = filename + "_conv.txt"
np.savetxt(input_filename, input_data, fmt='%ld\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')

print("Grid (1D) created with", ncells, "grid points!")
