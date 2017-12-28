import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys



# ----------------------------------------------
#    Create a random grid to test ray_tracing
# ----------------------------------------------



def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin



ncells = int(sys.argv[1])

np.random.seed(9001)

x = randrange(ncells,-10,10)
y = randrange(ncells,-10,10)
z = randrange(ncells,-10,10)

# Place the first point in the origin for convenience in testing

x[0] = 0.000000
y[0] = 0.000000
z[0] = 0.000000

x[1] = 0.100000
y[1] = 0.070000
z[1] = 0.130000

x[2] = -0.170000
y[2] = 0.070000
z[2] = -0.130000

x[2] = 0.140000
y[2] = -0.050000
z[2] = -0.180000


vx = np.zeros(ncells)
vy = np.zeros(ncells)
vz = np.zeros(ncells)


density = 10*np.ones(ncells)


data = np.stack((x, y, z, vx, vy, vz, density), axis=1)
np.savetxt('files/grid.txt', data, fmt='%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')

print("Grid created with", ncells, "grid points!")


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
