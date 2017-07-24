import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# ----------------------------------------------
#    Create a random grid to test ray_tracing
# ----------------------------------------------



def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin



ngrid = 100 #int(input("How many grid points?"))

np.random.seed(9001)

x = randrange(ngrid,-10,10)
y = randrange(ngrid,-10,10)
z = randrange(ngrid,-10,10)

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


vx = np.zeros(ngrid)
vy = np.zeros(ngrid)
vz = np.zeros(ngrid)


data = np.stack((x, y, z, vx, vy, vz), axis=1)
np.savetxt('grid.txt', data, fmt='%lf\t%lf\t%lf\t%lf\t%lf\t%lf')

print("Grid created with", ngrid, "grid points!")


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
