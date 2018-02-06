import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


# --------------------------------------------
# Create a regural 2D grid to test ray tracing
# --------------------------------------------


# ncells = int(sys.argv[1])
ncells = 6


ID = [ i for i in range(ncells)]

x  = [0, 1, 2, 0, 1, 2]#[ i/3 for i in range(ncells)]
y  = [0, 0, 0, 1, 1, 1]#[ i%2 for i in range(ncells)]
z  = np.zeros(ncells)

vx = np.zeros(ncells)
vy = np.zeros(ncells)
vz = np.zeros(ncells)

density = np.ones(ncells)


data = np.stack((ID, x, y, z, vx, vy, vz, density), axis=1)
np.savetxt('files/tests/grid_2D_test_6.txt', data, fmt='%ld\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')

print("Grid (2D) created with", ncells, "grid points!")

# Plot the resulting grid

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, z)
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")

#fig.tight_layout()

#plt.show()
#fig.savefig("plots.pdf", bbox_inches='tight')
