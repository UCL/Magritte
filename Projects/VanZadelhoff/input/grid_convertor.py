# Convert grid of Van Zadelhoff et al. to Magritte input grid

import numpy as np


# Read data in Van Zadelhoff format

data = np.loadtxt('files/model_1.d', skiprows=7, unpack=True)

ncells = np.shape(data)[1]

x           = data[0,:]
y           = np.zeros(ncells)
z           = np.zeros(ncells)
density     = data[1,:]
abundance   = data[2,:]
temperature = data[3,:]
vx          = data[4,:]
vy          = np.zeros(ncells)
vz          = np.zeros(ncells)
v_turb      = data[5,:]

zeros = np.zeros(ncells)
ones  = np.ones(ncells)

grid = np.stack((x, y, z, vx, vy, vz, density), axis=1)

abundance = abundance * density

abun = np.stack((zeros, abundance, density, abundance, ones), axis=1)

np.savetxt('files/grid.txt', grid, fmt='%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')
np.savetxt('files/temperature_gas.txt', temperature, fmt='%lE')
np.savetxt('files/abundances.txt', abun, fmt='%lE\t%lE\t%lE\t%lE\t%lE')
