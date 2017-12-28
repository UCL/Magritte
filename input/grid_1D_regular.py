import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys



# ----------------------------------------
# Create a 1D grid to test Ray Tracing
# ----------------------------------------



def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin


# Get the desired number of cells form the argument

ncells = int(sys.argv[1])


np.random.seed(9001)

x = 0.527046*np.array(range(ncells))
y = 0.527046*np.array(range(ncells))
z = 0.666667*np.array(range(ncells))

vx = np.zeros(ncells)
vy = np.zeros(ncells)
vz = np.zeros(ncells)

density = np.ones(ncells)

data = np.stack((x, y, z, vx, vy, vz, density), axis=1)
filename = "files/1D_regular_" + str(ncells) + ".txt"
np.savetxt(filename, data, fmt='%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')

print("Grid (1D) created with", ncells, "grid points!")

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
